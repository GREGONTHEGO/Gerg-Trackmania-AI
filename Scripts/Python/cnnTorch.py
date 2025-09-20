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


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
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
class Conv2DDDQNAgent(nn.Module):
    def __init__(self):
        super(Conv2DDDQNAgent, self).__init__()
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

        # self.move = nn.Linear(256 + 64,3)
        self.turn = nn.Linear(256 + 64, 9)
        # self.critic = nn.Sequential( 
        #     nn.Linear(256 + 64, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64,1)
        #     )
        # self.init_weights()



    def forward(self, x, telem):
        print(x.shape)
        B, T, C, H, W = x.shape
        x = torch.reshape(x,(B * T, C, H, W))
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
        # move_logits = self.move(joint)
        turn_logits = self.turn(joint)
        action = f.softmax(turn_logits, dim = 1)
        # critic = self.critic(joint).squeeze(-1)
        return action #critic move_logits, 
    


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
        # nn.init.constant_(self.move.weight, 0)
        # nn.init.constant_(self.move.bias, 0)
        nn.init.constant_(self.turn.weight, 0)
        nn.init.constant_(self.turn.bias, 0)
        # nn.init.kaiming_normal_(self.critic[0].weight)
        # nn.init.constant_(self.critic[0].bias, 0) 


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, image, action, reward, next_image, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, image, action, reward, next_image, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, images, actions, rewards, next_images, next_states, dones = [], [], [], [], [], [], []
        for idx in batch:
            state, image, action, reward, next_image, next_state, done = self.buffer[idx]
            states.append(state)
            images.append(image.squeeze(0))
            actions.append(action)
            rewards.append(reward)
            next_images.append(next_image.squeeze(0))
            next_states.append(next_state)
            dones.append(done)

        return (
            torch.tensor(np.array(states), dtype=torch.float32).squeeze(1).to(device),
            torch.tensor(np.stack(images), dtype=torch.float32).to(device),
            torch.tensor(np.array(actions), dtype=torch.int64).to(device),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.stack(next_images), dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).squeeze(1).to(device),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)
    
class DDQNAgent:
    def __init__(self, state_size, action_size, seed, learning_rate=1e-3, capacity=10000, discount_factor=0.99, tau=1e-3, update_every=4, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0

        self.qnetwork_local = Conv2DDDQNAgent().to(device)
        self.qnetwork_local.init_weights()
        self.qnetwork_target = Conv2DDDQNAgent().to(device)
        self.qnetwork_target.init_weights()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(capacity)
        self.update_target_network()

    def step(self, state, image, action, reward, next_image, next_state, done):
        self.replay_buffer.push(state, image, action, reward, next_image, next_state, done)

        self.steps += 1
        if self.steps % self.update_every == 0 and len(self.replay_buffer) >= self.batch_size:
            experiences = self.replay_buffer.sample(self.batch_size)
            self.learn(experiences)

    def act(self,image, state, eps=0.05):
        imag = torch.from_numpy(image).float().to(device)
        state = torch.from_numpy(state).float().to(device) #check if state is in this format
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(imag,state)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        states, images, actions, rewards, next_images, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_images, next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + self.discount_factor * (Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(images, states).gather(1, actions.view(-1, 1))

        loss = f.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def update_target_network(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    

def compute_reward(prev_state, current_state, alpha=1.0): # move
    speed = current_state['speed'] # in m/s
    prev_speed = prev_state['speed'] # in m/s
    # initiates reward
    reward = 0.0

    # if the current speed is positive give a big reward scaled by that speed
    if speed > 0.5:
        reward += alpha * speed

    # penalize if the vehicle is not moving
    # if move[0] < 0 and move[1] < 0 and abs(current_state['x'] - prev_state['x']) < 0.03 and abs(current_state['z'] - prev_state['z']) < 0.03:
    #     reward -= 3.0

    # # penalize if the car is breaking at low speed
    # if move[1] > 0 and speed < 5.0:
    #     reward -= 10.0 * (5.0 - speed)

    # penalize slow driving as the cp state rarely changes
    if current_state['cp'] == prev_state['cp'] and speed < 5.0:
        reward -= 1.0 * (5.0 - speed)

    # if the acceleration is positive and speed is positive give reward
    delta_spd = speed - prev_speed
    if delta_spd > 0.05 and speed > 1.0:
        reward += 1.0 * abs(delta_spd)

    # if decelerating and speed is positive give penalty
    if (delta_spd) < -0.05 and speed > 0.0:
        reward -= 1.0 * abs(delta_spd)

    return reward


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
    prev_action = 8
    prev_stacked = np.stack([f[0] for f in frame_buffer], axis=0)
    prev_stacked = np.expand_dims(prev_stacked, 1)
    prev_stacked = np.expand_dims(prev_stacked, 0)
    # to cut the system short if it doesn't make it to a new cp
    cp_time = time.time()

    # current cp num to see when we progress
    cp_num = 0

    # if stuck for a second with minimal movement break
    stuck_count = 0

    # if crossed the finish line end
    end_count = 0

    score = 0
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
        prev_state_vec = np.array([[prev_state['speed'], prev_state['x'], prev_state['y'], prev_state['z'], prev_state['cp']]])
        stacked = np.expand_dims(stacked, 1)
        stacked = np.expand_dims(stacked, 0)

        if abs(state['x'] - prev_state['x']) < 0.03 and abs(state['z'] - prev_state['z']) < 0.03:
            stuck_count += 1
        else:
            stuck_count = 0

        # runs the information into the model to get outputs
        # with torch.no_grad():
        #     action = model(torch.from_numpy(stacked).to(device), torch.tensor(state_vec, dtype=torch.float32).to(device))
        action = model.act(stacked, state_vec)
        reward = compute_reward(prev_state, state, cp_time)

        model.step(prev_state_vec, stacked, prev_action, reward, prev_stacked, state_vec, 0)

        
        if stuck_count > 30:
            reward -= 5.0

        score += reward

        prev_state = state.copy()
        prev_action = action.copy()
        prev_stacked = stacked.copy()
        # transition = {
        #     'state': state_vec[0],
        #     'image': stacked,
        #     'move': action,
        #     'reward': reward,
        #     'cp': state['cp']
        # }

        # transitions_since_last_cp.append(transition)

        # if a new cp is reached give a reward to the states that made it get to the new cp divided by how many states
        # if state['cp'] > cp_num:
        #     cp_reward = min(100.0 / len(transitions_since_last_cp), 5.0)
        #     for t in transitions_since_last_cp:
        #         t['reward'] += cp_reward
        #     episode_data.extend(transitions_since_last_cp)
        #     transitions_since_last_cp = []
        #     cp_num = state['cp']
        #     cp_time = time.time()

        # completes the action determined above
        
        if action == 0:
            keyboard.press('w')
            keyboard.release('a')
            keyboard.release('s')
            keyboard.release('d')
        elif action == 1:
            keyboard.press('w')
            keyboard.press('a')
            keyboard.release('s')
            keyboard.release('d')
        elif action == 2:
            keyboard.press('w')
            keyboard.release('a')
            keyboard.release('s')
            keyboard.press('d')
        elif action == 3:
            keyboard.release('w')
            keyboard.release('a')
            keyboard.press('s')
            keyboard.release('d')
        elif action == 4:
            keyboard.release('w')
            keyboard.press('a')
            keyboard.press('s')
            keyboard.release('d')
        elif action == 5:
            keyboard.release('w')
            keyboard.release('a')
            keyboard.press('s')
            keyboard.press('d')
        elif action == 6:
            keyboard.release('w')
            keyboard.release('a')
            keyboard.release('s')
            keyboard.release('d')
        elif action == 7:
            keyboard.release('w')
            keyboard.press('a')
            keyboard.release('s')
            keyboard.release('d')
        elif action == 8:
            keyboard.release('w')
            keyboard.release('a')
            keyboard.release('s')
            keyboard.press('d')

        
        time.sleep(1.0/60.0)

    # if there are actions since the last cp was reached add them to the end of the data
    # if transitions_since_last_cp:
    #     episode_data.extend(transitions_since_last_cp)

    # if running into a wall at the beginning of the map ignore the run
    # if stuck_count >= 60 and cp_num == 0:
    #     episode_data = []

    # if end_count >= 15:
    #     for reward in episode_data:
    #         reward['reward'] += 1000.0 / len(episode_data) if episode_data else 1.0

    # make sure that every button is released and restart the run
    keyboard.release('w')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    keyboard.press(Key.delete)
    keyboard.release(Key.delete)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)

    # total_reward = sum(t['reward'] for t in episode_data)/ len(episode_data) if episode_data else 1.0
    return episode_data, cp_num,# total_reward

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
    # collected_data = []
    for i in range(1):
        data, cp = run_episode(model, action_stats, greedy=False)
    #     if not data:
    #         print(f"Episode {i} ended in wall, skipping")
    #         continue
    #     collected_data.extend(data)
    #     update_best_episode(data, cp, reward)
    #     print(f"Episode {i} ended with CP {cp}, Reward {reward:.2f}")

    # if best_episode_data is not None:
    #     collected_data.extend(best_episode_data)

    # if collected_data:
    #     with open(f'{0}statesInEpoch.pkl', 'wb') as f:
    #         pickle.dump(collected_data, f)
    #     print(f"Episode {0} data saved with {len(collected_data)} transitions.")

        
def main():
    init_camera()
   
    model = DDQNAgent(5, 9, 5)
    # optimizer = optim.Adam([{'params':model.move.parameters()},{'params':model.turn.parameters()}, {'params':model.critic.parameters(), 'lr': 5e-6}], lr=5e-5)

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
        # model = train(model, optimizer).to(device)

    end_event.set()
    stop_capture.set()
    cap_thread.join()
    camera.stop()
    read.join()

if __name__ == "__main__":
    main()
