from pynput.keyboard import Key, Controller
import time
import os
import math
import socket
from threading import Thread
import threading
import numpy as np
import pickle
import dxcam, cv2
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader

BUFFER_SIZE = 10
CAPTURE_FPS = 6
.0
frame_buffer = deque(maxlen=BUFFER_SIZE)

region = (640, 300, 1920, 1200)
# print(dxcam.device_info())
camera = None

state_lock = threading.Lock()

HOST = '127.0.0.1'
PORT = 5055
keyboard = Controller()
shutdown_event = threading.Event()
end_event = threading.Event()
connection_event = threading.Event()
stop_capture = threading.Event()

latest_state = {'ts':0,'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'cp': 0.0}
episode = 0

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# print("torch version:", torch.__version__)
# print("CUDA support compiled:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())
# print("CUDA available:", torch.cuda.is_available())
# print("GPU count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("GPU name:", torch.cuda.get_device_name(0))
print(device)

def init_camera():
    global camera
    camera = dxcam.create(output_idx=0, output_color="GRAY", region=region, max_buffer_len=10)
    camera.start(target_fps=60)

def get_screenshot():
    screenshot = camera.get_latest_frame()
    img = screenshot[:, :, :1]  
    img = cv2.resize(img, (200, 100))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def capture_loop():
    interval = 1.0/CAPTURE_FPS
    while not stop_capture.is_set():
        img = get_screenshot()
        frame_buffer.append(img)
        time.sleep(interval)


class Conv2DAgent(nn.Module):
    def __init__(self):
        super(Conv2DAgent, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=(5,5), padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(256,256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.telem_norm =nn.LayerNorm(5)
        self.telemetry_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.move = nn.Linear(256 + 64,3)
        self.turn = nn.Linear(256 + 64,3)
        self.critic = nn.Linear(256 + 64, 1)
        # self.init_weights()

    def forward(self, x, telem):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = f.relu(self.bn1(self.conv2d_1(x)))
        x = f.relu(self.bn2(self.conv2d_2(x)))
        x = f.relu(self.bn3(self.conv2d_3(x)))
        x = f.relu(self.bn4(self.conv2d_4(x)))
        x = self.pool(x)
        x = x.view(B * T, -1)
        x = f.relu(self.fc1(x))
        x = x.view(B, T, -1)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        telem = self.telem_norm(telem)
        t_feat = self.telemetry_mlp(telem)
        joint = torch.cat([h_last, t_feat], dim=1)
        move_logits = self.move(joint)
        turn_logits = self.turn(joint)
        critic = self.critic(joint).squeeze(-1)
        return move_logits, turn_logits, critic
    
    # def init_weights(self):

    

def train_step(model, optimizer, images, tel, move, turn, returns):
    model.train()
    images = images.to(device)
    move = move.to(device)
    turn = turn.to(device)
    tel = tel.to(device)
    optimizer.zero_grad()
    move_logits, turn_logits, value = model(images, tel)
    adv = compute_advantage(returns, value)
    # ret_mean = returns.mean()
    # ret_std = returns.std()
    # returns = (returns - ret_mean) / (ret_std + 1e-8)
    returns = returns.to(device)
    adv = adv.to(device)
    # move_loss = f.cross_entropy(move_logits, move, reduction='none')
    # turn_loss = f.cross_entropy(turn_logits, turn, reduction='none')
    # update actor and critic loss
    logp_move = f.log_softmax(move_logits, dim=-1)
    logp_turn = f.log_softmax(turn_logits, dim=-1)

    actor_loss_move = -logp_move.gather(1, move.unsqueeze(1)).squeeze(1) * adv
    actor_loss_turn = -logp_turn.gather(1, turn.unsqueeze(1)).squeeze(1) * adv
    actor_loss = (actor_loss_move + actor_loss_turn).mean()

    critic_loss = f.mse_loss(value, returns)

    entropy_move = -(logp_move * f.softmax(move_logits, dim=-1)).sum(dim=1).mean()
    entropy_turn = -(logp_turn * f.softmax(turn_logits, dim=-1)).sum(dim=1).mean()
    entropy = 0.01 * (entropy_move + entropy_turn)
    print('actor ', actor_loss, ' critic ', critic_loss)
    loss = actor_loss + 0.5 * critic_loss - entropy
    loss.backward()
    optimizer.step()
    return loss.item()

class TrackmaniaDataset(Dataset):
    def __init__(self, states):
        self.images = np.concatenate([state['image'] for state in states], axis=0)
        self.stat = np.stack([state['state'] for state in states], axis=0).astype(np.float32)
        self.move = np.stack([state['move'] for state in states], axis=0).astype(np.int64)
        self.turn = np.stack([state['turn'] for state in states], axis=0).astype(np.int64)
        self.rewards = np.array([state['reward'] for state in states], dtype=np.float32)
        
    
    def __len__(self):
        return len(self.move)

    def __getitem__(self, idx):
        return(
            torch.from_numpy(self.images[idx]),
            self.stat[idx],
            self.move[idx],
            self.turn[idx],
            self.rewards[idx]
        )
    
def heading_reward(current_state, prev_state, cp_positions, k_heading=1.0):
    dx = current_state['x'] - prev_state['x']
    dz = current_state['z'] - prev_state['z']
    if dx==0 and dz==0:
        return 0.0
    actual_heading = math.atan2(dz, dx)

    next_cp = int(current_state['cp']) + 1
    if next_cp in cp_positions:
        x_cp, z_cp = cp_positions[next_cp]['x'], cp_positions[next_cp]['z']
        desired_heading = math.atan2(z_cp - current_state['z'], x_cp - current_state['x'])
    else:
        desired_heading = actual_heading

    err = (actual_heading - desired_heading + math.pi) % (2*math.pi) - math.pi
    return  k_heading * (1.0 - abs(err)) / math.pi


def proximity_reward(current_state, prev_state, cp_positions, k_prox=5.0):
    next_cp = int(current_state['cp']) + 1
    if next_cp not in cp_positions:
        return 0.0
    
    cp_x = cp_positions[next_cp]['x']
    cp_z = cp_positions[next_cp]['z']
    def dist(x, z): return math.sqrt((x - cp_x)**2 + (z - cp_z)**2)

    prev_dist = dist(prev_state['x'], prev_state['z'])
    cur_dist = dist(current_state['x'], current_state['z'])

    delta = prev_dist - cur_dist

    return k_prox * delta


def compute_reward(prev_state, current_state, move, alpha=0.1):
    speed = current_state['speed']
    prev_speed = prev_state['speed']
    # cp_diff = (current_state['cp'] - prev_state['cp'])
    # cp_reward = gamma * 10.0 * max(cp_diff, 0)

    reward = alpha * max(0.0, speed) #+ cp_reward
    if move == 2 and abs(current_state['x'] - prev_state['x']) < 0.1 and abs(current_state['z'] - prev_state['z']) < 0.1:
        reward -= 5.0
    if move == 1 and speed < 0.0:
        reward -= 5.0

    delta_spd = speed - prev_speed
    if (delta_spd) < -5.0 and speed > 0.0:
        reward -= 10.0 * abs(delta_spd)
    
    # if move == 0:
    #     reward += 0.1
    return reward

def compute_advantage(rewards, values, gamma=0.99):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        advantages[t] = running_add
    advantages = torch.tensor(advantages).to(device)
    advantages = advantages - values
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    return advantages

def train(model, optimizer, cp_positions, init=False):
    BATCH_SIZE = 16
    def train_batch(dataset):
        model.train()
        loaders = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
        total_loss = 0.0
        for images,tel, move,  turn,  rewards in loaders:
            loss = train_step(model, optimizer, images, tel, move, turn, rewards)
            total_loss += loss
            # print(loss)
        #print(f"Avg Loss: {total_loss/len(loaders):.4f}")
        # check to see if it is necessary to rerun the data through the model in training


    if init:
        for _ in range(10):
            for i in ['turn.pkl']: #,'straight.pkl', 'long.pkl''curve.pkl' , 'circle.pkl'
                #print(i)
                with open(f'0{i}', 'rb') as f:
                    statesInEpoch = pickle.load(f)
                dataset = TrackmaniaDataset(statesInEpoch)
                train_batch(dataset)
    else:
        for ep in range(episode):
            #print(ep)
            with open(f'{ep}statesInEpoch.pkl', 'rb') as f:
                statesInEpoch = pickle.load(f)
            dataset = TrackmaniaDataset(statesInEpoch)
            train_batch(dataset)
        
    # model.save('model_weights.keras', save_format='keras')
    return model

def read_game_state():
    global latest_state
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print("listening on", HOST, PORT)
        conn, addr = s.accept()
        connection_event.set()
        print(f"connection from {addr}")
        while not end_event.is_set():
            while not shutdown_event.is_set():
                time.sleep(0.001)
            try:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break
                parts = data.split('\n')
                if parts[-1] == '':
                    last_line = parts[-2]
                else:
                    last_line = parts[-1]
                fields = last_line.split(',')
                if len(fields) == 6 and all(fields):
                    ts, forwardVel, x, y, z, cp = fields
                    with state_lock:
                        latest_state['ts'] = int(ts)
                        latest_state['speed'] = float(forwardVel)
                        latest_state['x'] = float(x)
                        latest_state['y'] = float(y)
                        latest_state['z'] = float(z)
                        latest_state['cp'] = float(cp)
            except Exception as e:
                print("Error reading data:", e)
                break
        conn.close()

def inference(model, cp_positions):
    global episode
    episode = 0
    for i in range(1):
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        #print(i)
        start_time = time.time()
        episode_data = []
        with state_lock:
            prev_state = latest_state.copy()
        cp_time = time.time()
        cp_num = 0
        while (time.time() - start_time) < 30 and (time.time() - cp_time) < 10:
            if time.time() - start_time < 2:
                cp_time = time.time() + 2
                time.sleep(2)
            # start = time.perf_counter()
            with state_lock:
                state = latest_state.copy()
                stacked = np.stack([f[0] for f in frame_buffer], axis=0)
            if state['ts'] == prev_state['ts']:
                time.sleep(1.0/60.0)
                continue
            if state['cp'] > cp_num:
                if state['cp'] in cp_positions:
                    cur_time = time.time() - start_time
                    if cp_positions[state['cp']]['time'] > cur_time:
                        cp_positions[state['cp']] = {'time': cur_time, 'x': state['x'], 'z': state['z']}
                else:
                    cur_time = time.time() - start_time
                    cp_positions[state['cp']] = {'time': cur_time, 'x': state['x'], 'z': state['z']}
                cp_num = state['cp']
                cp_time = time.time()
            state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['cp']]])
            stacked = np.expand_dims(stacked, 1)
            stacked = np.expand_dims(stacked, 0)

            with torch.no_grad():
                move_log, turn_log, _ = model(torch.from_numpy(stacked).to(device), torch.tensor(state_vec, dtype=torch.float32).to(device))
                probs_move = torch.softmax(move_log[0], dim=0).cpu().numpy()
                probs_turn = torch.softmax(turn_log[0], dim=0).cpu().numpy()
            # epsilon = 0.05
            # rand = np.random.choice([0,1], p=[epsilon, 1-epsilon])

            # if rand == 1:
            #     move_action = np.argmax(probs_move)
            #     turn_action = np.argmax(probs_turn)
            # else:
            move_action = np.random.choice([0, 1, 2], p=probs_move)
            turn_action = np.random.choice([0, 1, 2], p=probs_turn)

            hr = heading_reward(state, prev_state, cp_positions)
            # print('err ',hr)
            dr = proximity_reward(state, prev_state, cp_positions)
            # print('prox ', dr)
            reward = compute_reward(prev_state, state, move_action)
            # print('reward ',reward, 'state ', state)
            reward += hr
            reward += dr
            # if move_action != 2:
            #     reward += 0.2
            # print('reward ',reward)
            prev_state = state.copy()
            episode_data.append({'state':state_vec[0], 'image': stacked,'move': move_action, 'turn': turn_action,'reward': reward})
            
            if move_action == 0:
                keyboard.press('w')
            else:
                keyboard.release('w')
            if move_action == 1:
                keyboard.press('s')
            else:
                keyboard.release('s')
            if turn_action == 0:
                keyboard.press('a')
            else:
                keyboard.release('a')
            if turn_action == 1:
                keyboard.press('d')
            else:
                keyboard.release('d')
            # dur = (time.perf_counter() - start) * 1000
            # print(f'[TIMING] Inference took {dur:.1f} ms')
            time.sleep(1.0/60.0)
        with open(f'{episode}statesInEpoch.pkl', 'wb') as f:
            pickle.dump(episode_data, f)

        episode += 1
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
    
def main():
    init_camera()
    CP_POSITIONS_FILE = 'cp_positions.pkl'
    if os.path.exists(CP_POSITIONS_FILE):
        with open(CP_POSITIONS_FILE, 'rb') as f:
            cp_positions = pickle.load(f)
    else:
        cp_positions = {}
    model = Conv2DAgent().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    read = Thread(target=read_game_state)
    read.start()
    connection_event.clear()
    connection_event.wait()
    for _ in range(BUFFER_SIZE):
        frame_buffer.append(get_screenshot())
    cap_thread = Thread(target=capture_loop)
    cap_thread.start()
    # model = train(model, optimizer, cp_positions, True).to(device)
    for i in range(1000):
        print(i)
        shutdown_event.set()
        inf = Thread(target=inference, args=(model,cp_positions))
        inf.start()
        inf.join()
        shutdown_event.clear()
        model = train(model, optimizer, cp_positions).to(device)
    end_event.set()
    stop_capture.set()
    cap_thread.join()
    camera.stop()
    read.join()

if __name__ == "__main__":
    main()