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
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader

BUFFER_SIZE = 5
CAPTURE_FPS = 60.0
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
best_episode_data = None
best_cp = -1
best_reward = -float('inf')



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
        self.conv2d_1 = nn.Conv2d(1, 64, kernel_size=(5,5), padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv2d_3 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv2d_4 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(512,256)
        # self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.telem_norm =nn.LayerNorm(5)
        self.telemetry_mlp = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.move = nn.Linear(256 + 64,3)
        self.turn = nn.Linear(256 + 64,3)
        self.critic = nn.Sequential( 
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64,1)
            )
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
        # _, (h_n, _) = self.lstm(x)
        # h_last = h_n[-1]
        h_last = x.mean(dim=1)
        telem = self.telem_norm(telem)
        t_feat = self.telemetry_mlp(telem)
        joint = torch.cat([h_last, t_feat], dim=1)
        move_logits = self.move(joint)
        turn_logits = self.turn(joint)
        critic = self.critic(joint).squeeze(-1)
        return move_logits, turn_logits, critic
    


    def init_weights(self):
        def init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(init)
        nn.init.constant_(self.move.weight, 0)
        nn.init.constant_(self.move.bias, 0)
        nn.init.constant_(self.turn.weight, 0)
        nn.init.constant_(self.turn.bias, 0)
        nn.init.kaiming_normal_(self.critic[0].weight)
        nn.init.constant_(self.critic[0].bias, 0) 

    

def train_step(model, optimizer, images, tel, move, turn, rewards):
    model.train()
    images = images.to(device)
    move = move.to(device)
    turn = turn.to(device)
    tel = tel.to(device)
    optimizer.zero_grad()
    move_logits, turn_logits, value = model(images, tel)
    adv, returns = compute_advantage(rewards, value)
    # ret_mean = returns.mean()
    # ret_std = returns.std()
    # returns = (returns - ret_mean) / (ret_std + 1e-8)
    returns = returns.to(device)
    # print(f"Return mean: {returns.mean():.4f}, std: {returns.std():.4f}")
    # print(f"Value mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
    # print("value sample:", value[:5].cpu().detach().numpy())
    adv = adv.to(device)
    # move_loss = f.cross_entropy(move_logits, move, reduction='none')
    # turn_loss = f.cross_entropy(turn_logits, turn, reduction='none')
    # update actor and critic loss
    logp_move = f.log_softmax(move_logits, dim=-1)
    logp_turn = f.log_softmax(turn_logits, dim=-1)

    actor_loss_move = -logp_move.gather(1, move.unsqueeze(1)).squeeze(1) * adv
    actor_loss_turn = -logp_turn.gather(1, turn.unsqueeze(1)).squeeze(1) * adv
    actor_loss = (actor_loss_move + actor_loss_turn).mean()

    critic_loss = f.smooth_l1_loss(value, returns)

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
        # print('reward range: ', np.min(self.rewards), np.max(self.rewards))
        
    
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
    
# def heading_reward(current_state, prev_state, cp_positions, k_heading=1.0):
#     dx = current_state['x'] - prev_state['x']
#     dz = current_state['z'] - prev_state['z']
#     if dx==0 and dz==0:
#         return 0.0
#     actual_heading = math.atan2(dz, dx)

#     next_cp = int(current_state['cp']) + 1
#     if next_cp in cp_positions:
#         x_cp, z_cp = cp_positions[next_cp]['x'], cp_positions[next_cp]['z']
#         desired_heading = math.atan2(z_cp - current_state['z'], x_cp - current_state['x'])
#     else:
#         desired_heading = actual_heading

#     err = (actual_heading - desired_heading + math.pi) % (2*math.pi) - math.pi
#     return  k_heading * (1.0 - abs(err)) / math.pi


# def proximity_reward(current_state, prev_state, cp_positions, k_prox=5.0):
#     next_cp = int(current_state['cp']) + 1
#     if next_cp not in cp_positions:
#         return 0.0
    
#     cp_x = cp_positions[next_cp]['x']
#     cp_z = cp_positions[next_cp]['z']
#     def dist(x, z): return math.sqrt((x - cp_x)**2 + (z - cp_z)**2)

#     prev_dist = dist(prev_state['x'], prev_state['z'])
#     cur_dist = dist(current_state['x'], current_state['z'])

#     delta = prev_dist - cur_dist

#     return k_prox * delta



def compute_reward(prev_state, current_state, move, tim, alpha=1.0):
    speed = current_state['speed']
    prev_speed = prev_state['speed']
    # cp_diff = (current_state['cp'] - prev_state['cp'])
    # cp_reward = gamma * 10.0 * max(cp_diff, 0)

    reward = alpha * max(0.0, speed) #+ cp_reward
    if move == 2 and abs(current_state['x'] - prev_state['x']) < 0.1 and abs(current_state['z'] - prev_state['z']) < 0.1:
        reward -= 5.0
    if move == 1 and speed < 0.0:
        reward -= 5.0

    if prev_state['cp'] == 0.0 and speed > 5.0:
        reward += 5.0

    if current_state['cp'] == prev_state['cp'] and speed < 1.0:
        reward -= 2.0

    # if current_state['cp'] > prev_state['cp']:
    #     cp_time_delta = time.time() - tim
    #     reward += 100.0 / cp_time_delta

    delta_spd = speed - prev_speed
    if (delta_spd) < -5.0 and speed > 0.0:
        reward -= 10.0 * abs(delta_spd)
    
    # if move == 0:
    #     reward += 0.1
    return reward




def compute_advantage(rewards, values, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        returns[t] = running_add
    returns = torch.tensor(returns).to(device)
    advantages = returns - values.detach()
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    advantages = torch.clamp(advantages, -10.0, 10.0)
    # print(f'adv mean: {min(advantages):.4f}, std: {max(advantages):.4f}')
    return advantages, returns




def train(model, optimizer, cp_positions, init=False):
    BATCH_SIZE = 8
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
        for ep in range(1):
            #print(ep)
            with open(f'{ep}statesInEpoch.pkl', 'rb') as f:
                statesInEpoch = pickle.load(f)
            if not statesInEpoch:
                print(f'warning: Episode {ep} has no data, skipping')
                continue
            if len(statesInEpoch) == 0:
                print('no data')
                continue
            dataset = TrackmaniaDataset(statesInEpoch)
            train_batch(dataset)
        
    # model.save('model_weights.keras', save_format='keras')
    torch.save(model.state_dict(), 'model_cp.pt')
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




def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)



def inference(model,action_stats): # cp_positions, 
    # global episode # add thing to stop when the time stamp is the same for too long
    # smaller learning rate and smaller critic learning rate bigger temperature
    # episode = 0
    global best_episode_data, best_cp, best_reward
    for i in range(5):
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        #print(i)
        start_time = time.time()
        episode_data = []
        with state_lock:
            prev_state = latest_state.copy()
        cp_time = time.time()
        transitions_since_last_cp = []
        # last_cp_time = time.time()
        cp_num = 0
        stuck_count = 0
        cur_cp = -1
        cur_reward = 0.0
        stuck = False
        while (time.time() - start_time) < 240 and (time.time() - cp_time) < 15 and stuck_count < 60:
            if time.time() - start_time < 2:
                cp_time = time.time() + 2
                time.sleep(2)
            # start = time.perf_counter()
            with state_lock:
                state = latest_state.copy()
                stacked = np.stack([f[0] for f in frame_buffer], axis=0)
            if state['ts'] == prev_state['ts']:
                time.sleep(1.0/60.0)
                stuck_count += 1
                continue
            state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['cp']]])
            stacked = np.expand_dims(stacked, 1)
            stacked = np.expand_dims(stacked, 0)
            if abs(state['x'] - prev_state['x']) < 0.03 and abs(state['z'] - prev_state['z']) < 0.03:
                stuck_count += 1
            else:
                stuck_count = 0
            with torch.no_grad():
                move_log, turn_log, _ = model(torch.from_numpy(stacked).to(device), torch.tensor(state_vec, dtype=torch.float32).to(device))
                # probs_move = torch.softmax(move_log[0], dim=0).cpu().numpy()
                # probs_turn = torch.softmax(turn_log[0], dim=0).cpu().numpy()
                probs_move = softmax_with_temperature(move_log[0]).cpu().numpy() #, temperature=0.3
                probs_turn = softmax_with_temperature(turn_log[0]).cpu().numpy() #, temperature=0.3
            # epsilon = 0.05
            # rand = np.random.choice([0,1], p=[epsilon, 1-epsilon])

            # if rand == 1:
            # if i == 1:
            #     move_action = np.argmax(probs_move)
            #     turn_action = np.argmax(probs_turn)
            # else:
            move_action = np.random.choice([0, 1, 2], p=probs_move)
            turn_action = np.random.choice([0, 1, 2], p=probs_turn)

            # hr = heading_reward(state, prev_state, cp_positions)
            # # print('err ',hr)
            # dr = proximity_reward(state, prev_state, cp_positions)
            # # print('prox ', dr)
            reward = compute_reward(prev_state, state, move_action, cp_time)
            if stuck_count > 30:
                reward -= 5.0
            # print('reward ',reward, 'state ', state)
            # reward += hr
            # reward += dr
            # reward /= 10.0
            # if move_action != 2:
            #     reward += 0.2
            # print('reward ',reward)
            key = (move_action, turn_action)
            action_stats[key]['count'] += 1
            action_stats[key]['reward'] += reward
            prev_state = state.copy()
            transition = {
                'state': state_vec[0],
                'image': stacked,
                'move': move_action,
                'turn': turn_action,
                'reward': reward,
                'cp': state['cp']
            }
            transitions_since_last_cp.append(transition)
            # episode_data.append({'state':state_vec[0], 'image': stacked,'move': move_action, 'turn': turn_action,'reward': reward})
            if state['cp'] > cp_num:
                cp_reward = min(100.0 / len(transitions_since_last_cp), 5.0)
                # cp_contrib = [i for i, t in enumerate(transitions_since_last_cp) if t['cp'] == cp_num]
                for t in transitions_since_last_cp:
                    t['reward'] += cp_reward
                episode_data.extend(transitions_since_last_cp)
                transitions_since_last_cp = []
                cp_num = state['cp']
                cp_time = time.time()
                
                # if state['cp'] in cp_positions:
                #     cur_time = time.time() - start_time
                #     if cp_positions[state['cp']]['time'] > cur_time:
                #         cp_positions[state['cp']] = {'time': cur_time, 'x': state['x'], 'z': state['z']}
                # else:
                #     cur_time = time.time() - start_time
                #     cp_positions[state['cp']] = {'time': cur_time, 'x': state['x'], 'z': state['z']}
                # cp_num = state['cp']
                # cp_time = time.time()
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
        if transitions_since_last_cp:
            episode_data.extend(transitions_since_last_cp)
        # episode += 1
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        if cp_num > cur_cp or (cp_num == cur_cp and sum(t['reward'] for t in episode_data) > cur_reward):
            cur_cp = cp_num
            cur_reward = sum(t['reward'] for t in episode_data) / len(episode_data)
            if best_episode_data is None:
                best_episode_data = episode_data.copy()
                best_cp = cur_cp
                best_reward = cur_reward
            elif best_cp < cur_cp or (best_cp == cur_cp and best_reward < cur_reward):
                best_episode_data = episode_data.copy()
                best_cp = cur_cp
                best_reward = cur_reward
            print(f"New best episode: CP {best_cp}, Reward {best_reward:.2f}")
    
    with open(f'{episode}statesInEpoch.pkl', 'wb') as f:
        pickle.dump(best_episode_data, f)

        
def main():
    init_camera()
    CP_POSITIONS_FILE = 'cp_positions.pkl'
    if os.path.exists(CP_POSITIONS_FILE):
        with open(CP_POSITIONS_FILE, 'rb') as f:
            cp_positions = pickle.load(f)
    else:
        cp_positions = {}
    model = Conv2DAgent().to(device)
    model.init_weights()
    #model.load_state_dict(torch.load('best_so_far.pt'))
    optimizer = optim.Adam([{'params':model.move.parameters()},{'params':model.turn.parameters()}, {'params':model.critic.parameters(), 'lr': 1e-5}], lr=1e-4)
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
        action_stats = defaultdict(lambda: {'count': 0, 'reward': 0.0})
        shutdown_event.set()
        inf = Thread(target=inference, args=(model, action_stats)) #  cp_positions, 
        inf.start()
        inf.join()
        for (move, turn), stats in action_stats.items():
            print(f"Action ({move}, {turn}): Count = {stats['count']}, Total Reward = {stats['reward']/stats['count']:.2f}")
        shutdown_event.clear()
        model = train(model, optimizer, cp_positions).to(device)
    end_event.set()
    stop_capture.set()
    cap_thread.join()
    camera.stop()
    read.join()

if __name__ == "__main__":
    main()