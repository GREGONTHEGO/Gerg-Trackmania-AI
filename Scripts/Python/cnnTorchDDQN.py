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

"""
CNN with stacked grayscale images and telemetry (PyTorch)

This file defines a PyTorch CNN model that processes a 
stack of 5 grayscale game frames along with current 
telemetry values to predict movement actions using softmax 
heads. The model includes a critic for value estimation 
and is trained using an actor-critic approach. It improves 
efficiency by running on GPU, allows for checkpoint buffering, 
and supports entropy regularization. The model structure 
fuses spatial image features with telemetry via an MLP, and 
previously explored LSTM/3D convolution options are commented out.

"""

BUFFER_SIZE = 5
CAPTURE_FPS = 60.0
frame_buffer = deque(maxlen=BUFFER_SIZE)

region = (640, 300, 1920, 1200)
camera = None

state_lock = threading.Lock()

HOST = '127.0.0.1'
PORT = 5055
keyboard = Controller()

# used to put the read function to sleep
shutdown_event = threading.Event()

# used to stop reading from the game state thread
end_event = threading.Event()

# set to allow for the other threads to begin working once a connection to the game has occurred
connection_event = threading.Event()

# stops the camera
stop_capture = threading.Event()

# telemetry data used in the model and for the rewards
# speed, position and current cp
latest_state = {'ts':0,'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'cp': 0.0}

# connect to GPU otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# print(device)

# information to store the best run and some information about the best run
best_episode_data = None
best_cp = -1
best_reward = -float('inf')



def init_camera():
    global camera
    camera = dxcam.create(output_idx=0, output_color="GRAY", region=region, max_buffer_len=10)
    camera.start(target_fps=60)


# captures screen and as a grayscale and resizes it to be a smaller image in use in a CNN
def get_screenshot():
    screenshot = camera.get_latest_frame()
    img = screenshot[:, :, :1]  
    img = cv2.resize(img, (200, 100))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# collects images and buffers them in a frame_queue
def capture_loop():
    interval = 1.0/CAPTURE_FPS
    while not stop_capture.is_set():
        img = get_screenshot()
        frame_buffer.append(img)
        time.sleep(interval)


# model that has two main inputs: 5 grayscale images, and 5 telemetry data speed, position (x,y,z), and current cp
# outputs a critic, move and turn
# move uses softmax of 3 points to choose forward, backward or nothing
# turn does the same but for left, right or nothing
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

    
# Trains the softmax policy agent using previously logged data.
def train_step(model, optimizer, images, tel, move, turn, rewards):
    model.train()
    images = images.to(device)
    move = move.to(device)
    turn = turn.to(device)
    tel = tel.to(device)
    optimizer.zero_grad()
    move_logits, turn_logits, value = model(images, tel)
    adv, returns = compute_advantage(rewards, value)
    returns = returns.to(device)
    adv = adv.to(device)

    logp_move = f.log_softmax(move_logits, dim=-1)
    logp_turn = f.log_softmax(turn_logits, dim=-1)

    actor_loss_move = -logp_move.gather(1, move.unsqueeze(1)).squeeze(1) * adv
    actor_loss_turn = -logp_turn.gather(1, turn.unsqueeze(1)).squeeze(1) * adv
    actor_loss = (actor_loss_move + actor_loss_turn).mean()

    critic_loss = f.smooth_l1_loss(value, returns)

    entropy_move = -(logp_move * f.softmax(move_logits, dim=-1)).sum(dim=1).mean()
    entropy_turn = -(logp_turn * f.softmax(turn_logits, dim=-1)).sum(dim=1).mean()
    entropy = 0.01 * (entropy_move + entropy_turn)

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
    
"""Used to have heading and proximity rewards to guide the car to the next checkpoint if a car
previously hit that checkpoint by using the position of the car when the cp_count went up
but this did not account for the car hitting a checkpoint and being close to a wall or if there 
is a wall between the car and the next checkpoint"""
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


# Calculates scalar reward with comments explaining what each one should do.
def compute_reward(prev_state, current_state, move, tim, alpha=1.0):
    speed = current_state['speed']
    prev_speed = prev_state['speed']
    # cp_diff = (current_state['cp'] - prev_state['cp'])
    # cp_reward = gamma * 10.0 * max(cp_diff, 0)

    # if the current speed is positive give a small reward scaled by that speed
    reward = alpha * max(0.0, speed)

    # penalize if the vehicle is not moving
    if move == 2 and abs(current_state['x'] - prev_state['x']) < 0.1 and abs(current_state['z'] - prev_state['z']) < 0.1:
        reward -= 5.0

    # penalize if the car is breaking and going backwards
    if move == 1 and speed < 0.0:
        reward -= 5.0

    # give reward for driving fast at the start
    if prev_state['cp'] == 0.0 and speed > 5.0:
        reward += 5.0

    # penalize slow driving as the cp state rarely changes
    if current_state['cp'] == prev_state['cp'] and speed < 1.0:
        reward -= 2.0

    """originally had a bonus for passing a checkpoint but I realized that this would give the reward
    to only one move which would be problematic if it was braking when it crossed the checkpoint"""
    # if current_state['cp'] > prev_state['cp']:
    #     cp_time_delta = time.time() - tim
    #     reward += 100.0 / cp_time_delta

    # if sharply decelerating and speed is positive give penalty (supposed to be for hitting wall head on)
    delta_spd = speed - prev_speed
    if (delta_spd) < -5.0 and speed > 0.0:
        reward -= 10.0 * abs(delta_spd)
    
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

    return advantages, returns



# trains the model either from prior runs or from its own runs
def train(model, optimizer, init=False):
    BATCH_SIZE = 12
    def train_batch(dataset):
        model.train()
        loaders = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
        total_loss = 0.0
        for images,tel, move,  turn,  rewards in loaders:
            loss = train_step(model, optimizer, images, tel, move, turn, rewards)
            total_loss += loss

    # I had a recording function so that I could do some initial training on some of my runs
    # but I have dropped that since as I want the model to be able to learn how to drive from nothing
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



# reads the game state from the angelscript plugin that I made for trackmania
# connects to a port and will grab the most recent data point each frame
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


# this is the function that actually plays the game and records the states and all for training
def run_episode(model, action_stats, greedy=False):
    keyboard.press(Key.delete)
    keyboard.release(Key.delete)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)

    start_time = time.time()
    episode_data = []
    transitions_since_last_cp = []

    with state_lock:
        prev_state = latest_state.copy()

    # to cut the system short if it doesn't make it to a new cp
    cp_time = time.time()

    # current cp num to see when we progress
    cp_num = 0

    # if stuck for a second with minimal movement break
    stuck_count = 0

    # if crossed the finish line end
    end_count = 0

    while (time.time() - start_time) < 240 and (time.time() - cp_time) < 30 and stuck_count < 60 and end_count < 15:
        # waits for the map to actually start before giving or storing commands
        if time.time() - start_time < 2:
            cp_time = time.time() + 2
            time.sleep(2)
        
        with state_lock:
            state = latest_state.copy()
            stacked = np.stack([f[0] for f in frame_buffer], axis=0)
        
        if state['ts'] == prev_state['ts']:
            time.sleep(1.0/60.0)
            end_count += 1
            continue
        else:
            end_count = 0

        state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['cp']]])
        stacked = np.expand_dims(stacked, 1)
        stacked = np.expand_dims(stacked, 0)

        if abs(state['x'] - prev_state['x']) < 0.03 and abs(state['z'] - prev_state['z']) < 0.03:
            stuck_count += 1
        else:
            stuck_count = 0

        # runs the information into the model to get outputs
        with torch.no_grad():
            move_log, turn_log, _ = model(torch.from_numpy(stacked).to(device), torch.tensor(state_vec, dtype=torch.float32).to(device))
            probs_move = softmax_with_temperature(move_log[0]).cpu().numpy()
            probs_turn = softmax_with_temperature(turn_log[0]).cpu().numpy()

        """Was to add randomness into the runs but I just went to doing only random, as
        I figured it would learn more from having more different inputs"""
        # epsilon = 0.05
        # rand = np.random.choice([0,1], p=[epsilon, 1-epsilon])

        # if rand == 1:
        # if i == 1:
        #     move_action = np.argmax(probs_move)
        #     turn_action = np.argmax(probs_turn)
        # else:
        move_action = np.random.choice([0, 1, 2], p=probs_move)
        turn_action = np.random.choice([0, 1, 2], p=probs_turn)


        reward = compute_reward(prev_state, state, move_action, cp_time)
        if stuck_count > 30:
            reward -= 5.0

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

        # if a new cp is reached give a reward to the states that made it get to the new cp divided by how many states
        if state['cp'] > cp_num:
            cp_reward = min(100.0 / len(transitions_since_last_cp), 5.0)
            for t in transitions_since_last_cp:
                t['reward'] += cp_reward
            episode_data.extend(transitions_since_last_cp)
            transitions_since_last_cp = []
            cp_num = state['cp']
            cp_time = time.time()

        # completes the action determined above
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
        
        time.sleep(1.0/60.0)

    # if there are actions since the last cp was reached add them to the end of the data
    if transitions_since_last_cp:
        episode_data.extend(transitions_since_last_cp)

    # if running into a wall at the beginning of the map ignore the run
    if stuck_count >= 60 and cp_num == 0:
        episode_data = []

    if end_count >= 15:
        for reward in episode_data:
            reward['reward'] += 1000.0 / len(episode_data) if episode_data else 1.0

    # make sure that every button is released and restart the run
    keyboard.release('w')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    keyboard.press(Key.delete)
    keyboard.release(Key.delete)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)

    total_reward = sum(t['reward'] for t in episode_data)/ len(episode_data) if episode_data else 1.0
    return episode_data, cp_num, total_reward

# if there is a new best episode then update all of the global best episode info
def update_best_episode(episode_data, cp_num, cur_reward):
    global best_episode_data, best_cp, best_reward
    if cp_num > best_cp or (cp_num == best_cp and cur_reward > best_reward):
        best_episode_data = episode_data.copy()
        best_cp = cp_num
        best_reward = cur_reward
        print(f"New best episode: CP {best_cp}, Reward {best_reward:.2f}")

# compute five runs and then send that to a pkl file for the training to run out of
def inference(model,action_stats): 
    collected_data = []
    for i in range(5):
        data, cp, reward = run_episode(model, action_stats, greedy=False)
        if not data:
            print(f"Episode {i} ended in wall, skipping")
            continue
        collected_data.extend(data)
        update_best_episode(data, cp, reward)
        print(f"Episode {i} ended with CP {cp}, Reward {reward:.2f}")

    if best_episode_data is not None:
        collected_data.extend(best_episode_data)

    if collected_data:
        with open(f'{0}statesInEpoch.pkl', 'wb') as f:
            pickle.dump(collected_data, f)
        print(f"Episode {0} data saved with {len(collected_data)} transitions.")

        
def main():
    init_camera()
   
    model = Conv2DAgent().to(device)
    model.init_weights()
    optimizer = optim.Adam([{'params':model.move.parameters()},{'params':model.turn.parameters()}, {'params':model.critic.parameters(), 'lr': 5e-6}], lr=5e-5)

    read = Thread(target=read_game_state)
    read.start()

    connection_event.clear()
    connection_event.wait()

    for _ in range(BUFFER_SIZE):
        frame_buffer.append(get_screenshot())

    cap_thread = Thread(target=capture_loop)
    cap_thread.start()

    for i in range(1000):
        print(i)
        action_stats = defaultdict(lambda: {'count': 0, 'reward': 0.0})
        shutdown_event.set()
        inf = Thread(target=inference, args=(model, action_stats))
        inf.start()
        inf.join()
        for (move, turn), stats in action_stats.items():
            print(f"Action ({move}, {turn}): Count = {stats['count']}, Total Reward = {stats['reward']/stats['count']:.2f}")

        shutdown_event.clear()
        model = train(model, optimizer).to(device)

    end_event.set()
    stop_capture.set()
    cap_thread.join()
    camera.stop()
    read.join()

if __name__ == "__main__":
    main()
