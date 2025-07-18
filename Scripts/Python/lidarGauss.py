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
from torch.distributions import Normal

BUFFER_SIZE = 5
CAPTURE_FPS = 60.0
frame_buffer = deque(maxlen=BUFFER_SIZE)

region = (0, 100, 2560, 1400)

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
    camera = dxcam.create(output_idx=0, output_color="BGR", region=region, max_buffer_len=10)
    camera.start(target_fps=60)


# Converts screen image to high-contrast black/white mask.
def black_and_white_filter(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        mask = image < 50
        result = np.full_like(image, 255)
        result[mask] = 0
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black = gray < 50
    white = gray > 240
    result = np.full_like(image, 255)
    black_mask = np.repeat(black[:, :, np.newaxis], 3, axis=2)
    white_mask = np.repeat(white[:, :, np.newaxis], 3, axis=2)
    result[black_mask] = 0
    result[white_mask] = 255
    return result


# Captures screen, applies filter, returns processed frame.
# also reconnects camera if the latest frame is not recieved. Usually happens after around 8 hours of running
def get_screenshot():
    global camera
    screenshot = camera.get_latest_frame()
    if screenshot is None:
        print("[WARNING] No frame captured, retrying...")
        try:
            camera.stop()
        except:
            pass
        time.sleep(0.1)
        init_camera()
        screenshot = camera.get_latest_frame()
        if screenshot is None:
            raise RuntimeError("Failed to reinitialize camera.")
    img = black_and_white_filter(screenshot)
    return img


# Computes estimated distance from camera to object based on angle.
def pixel_to_distance(y, image_height=1600, vertical_fov=130, camera_tilt_degree=33, camera_height=1.5):
    pixel_from_bottom = abs(y)
    if pixel_from_bottom < 1:
        pixel_from_bottom = 0
    vertical_ratio = pixel_from_bottom / image_height
    beta_degree = vertical_fov * vertical_ratio
    total_angle = min(camera_tilt_degree + beta_degree, 89.7)
    total_angle_rad = math.radians(total_angle)
    distance = math.tan(total_angle_rad) * camera_height

    return distance


# Computes side distance using pixel X offset and known FOV.
def lateral_offset(x, forward_distance, image_width=2560, horizontal_fov=70):
    if x == 0:
        x = 1
    horizontal_ratio = x / image_width
    alpha_degree = horizontal_fov * horizontal_ratio
    alpha_rad = math.radians(alpha_degree)
    lateral_offset = forward_distance * math.tan(alpha_rad)

    return lateral_offset


# Casts multiple rays across the screen to find first obstacle in each direction.
def compute_dist(edges, origin, sin_angles, cos_angles, max_distance, W, H):
    num_rays = len(sin_angles)
    result = np.zeros((num_rays), dtype=np.float32)
    for i in range(num_rays):
        ray_mask = np.zeros((H, W), dtype=np.uint8)
        dx = (max_distance * sin_angles[i])
        dy = (max_distance * cos_angles[i])
        x = int(origin[0] + dx)
        y = int(origin[1] - dy)

        cv2.line(ray_mask, origin, (x, y), (255, 255, 255), 1)
        hit_mask = cv2.bitwise_and(edges, ray_mask)
        hit_points = cv2.findNonZero(hit_mask)

        if hit_points is not None:
            origin_np = np.array(origin, dtype=np.int32)
            distances = np.linalg.norm(hit_points[:, 0].astype(np.float32) - origin_np, axis=1)
            closest_index = np.argmin(distances)
            hit = tuple(hit_points[closest_index][0])
            dx_hit = hit[0] - origin[0]
            dy_hit = hit[1] - origin[1]

            fwd = pixel_to_distance(dy_hit, H)
            lat = lateral_offset(dx_hit, fwd, W)
            total = float(math.hypot(fwd, lat))
            result[i] = total

        else:
            fwd = pixel_to_distance(dy, H)
            lat = lateral_offset(dx, fwd, W)
            total = float(math.hypot(fwd, lat))
            result[i] = total

    return result


# Simulates LIDAR readings using image space ray intersection.
def lidar_numbers(image, num_rays=19, max_distance=1270):
    edges = cv2.Canny(image, 50, 150)
    H, W = edges.shape
    origin = (W // 2, H - 100)
    angles = np.linspace(-math.radians(90), math.radians(90), num_rays)
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)

    dist = compute_dist(edges, origin, sin_angles, cos_angles, max_distance, W, H)
    return dist


# Collects LIDAR scans continuously and buffers them in a frame queue.
def capture_loop():
    interval = 1.0/CAPTURE_FPS
    while not stop_capture.is_set():
        img = get_screenshot()
        frame_buffer.append(lidar_numbers(img))
        time.sleep(interval)


# model that has two main inputs: 19 log2 distances, and 5 telemetry data speed, position (x,y,z), and current cp
# outputs a critic, mu and log_std
# mu and log_std follow the same scheme where 0 is forward or nothing, 1 is bacward or nothing and 2 is left, nothing, or right
class LSTMAgent(nn.Module):
    def __init__(self):
        super(LSTMAgent, self).__init__()
        self.lidar_norm = nn.LayerNorm(19)
        self.lstm = nn.LSTM(input_size=19, hidden_size=2048, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(2048,1024)
        self.telem_norm =nn.LayerNorm(5)
        self.telemetry_mlp = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.joint_norm = nn.Linear(1024 + 256, 512)
        self.joint_layer = nn.Linear(512, 256)

        self.mu = nn.Linear(256,3)
        self.log_std = nn.Linear(256,3)

        self.critic = nn.Sequential( 
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1)
            )



    def forward(self, x, telem):
        LOG_STD_MIN = -20
        LOG_STD_MAX = 2
        x = self.lidar_norm(x) # start for the distance data
        lstm_out , _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        h_last = f.relu(self.fc1(x))
        telem = self.telem_norm(telem) # start for the telemetry data
        t_feat = self.telemetry_mlp(telem)
        joint = torch.cat([h_last, t_feat], dim=1) # data is joined here
        joint = self.joint_norm(joint)
        joint = self.joint_layer(joint)
        mu_logits = self.mu(joint)
        log_std = torch.clamp(self.log_std(joint), LOG_STD_MIN, LOG_STD_MAX)
        critic = self.critic(joint).squeeze(-1)

        return mu_logits, log_std, critic
    


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
        nn.init.constant_(self.mu.weight, 0)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.constant_(self.log_std.weight, 0)
        nn.init.constant_(self.log_std.bias, -1.0)
        nn.init.kaiming_normal_(self.critic[0].weight)
        nn.init.constant_(self.critic[0].bias, 0) 

    
# Trains the Gaussian policy agent using previously logged data.
def train_step(model, optimizer, images, tel, move, rewards):
    model.train()
    images = images.float().to(device)
    move = move.to(device)
    # turn = turn.to(device)
    tel = tel.to(device)

    optimizer.zero_grad()

    mu, std, value = model(images, tel)
    std = torch.exp(std).clamp(min=1e-6)
    dist = Normal(mu, std)
    action = dist.rsample()
    logp = dist.log_prob(action)
    logp = logp - torch.log(1 - torch.tanh(action).pow(2) + 1e-6)
    logp = logp.sum(dim=1)

    adv, returns = compute_advantage(rewards, value)
    returns = returns.to(device)
    adv = adv.to(device)

    critic_loss = f.smooth_l1_loss(value, returns)

    entropy = dist.entropy().sum(dim=-1).mean()

    actor_loss = -logp * adv

    loss = (actor_loss + 0.5 * critic_loss - 0.1 * entropy).mean()
    loss.backward()
    optimizer.step()
    return loss.item()



class TrackmaniaDataset(Dataset):
    def __init__(self, states):
        self.images = np.concatenate([state['image'] for state in states], axis=0)
        self.stat = np.stack([state['state'] for state in states], axis=0).astype(np.float32)
        self.move = np.stack([state['move'] for state in states], axis=0).astype(np.int64)
        self.rewards = np.array([state['reward'] for state in states], dtype=np.float32)
        
    def __len__(self):
        return len(self.move)

    def __getitem__(self, idx):
        return(
            torch.from_numpy(self.images[idx]),
            self.stat[idx],
            self.move[idx],
            self.rewards[idx]
        )


# Calculates scalar reward with comments explaining what each one should do.
def compute_reward(prev_state, current_state, move, alpha=1.0):
    speed = current_state['speed'] # in m/s
    prev_speed = prev_state['speed'] # in m/s
    # initiates reward
    reward = 0.0

    # if the current speed is positive give a big reward scaled by that speed
    if speed > 0.5:
        reward += alpha * speed

    # penalize if the vehicle is not moving
    if move[0] < 0 and move[1] < 0 and abs(current_state['x'] - prev_state['x']) < 0.03 and abs(current_state['z'] - prev_state['z']) < 0.03:
        reward -= 3.0

    # penalize if the car is breaking at low speed
    if move[1] > 0 and speed < 5.0:
        reward -= 10.0 * (5.0 - speed)

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


def lidar_clearance_bonus(delta_spd, move):

    # there was an issue that the car realized it got more rewards when decelerating as the camera angle would change and make the car seem further away
    # to prevent I added this if statement to not give a reward for decelerating
    if delta_spd < 0.0:
        return -1.0
    
    center_idk = 9
    bonus = 0.0

    for i in range(len(frame_buffer)):
        ray_distances = frame_buffer[i]
        forward_distance = ray_distances[center_idk]
        forward_left = np.mean(ray_distances[center_idk - 4: center_idk - 1])
        forward_right = np.mean(ray_distances[center_idk + 1: center_idk + 4])
        side_left = np.mean(ray_distances[:3])
        side_right = np.mean(ray_distances[-3:])

        # smallest forward distance
        forward_dist = min(forward_distance, forward_left, forward_right)

        # smallest side distance
        side_mean = min(side_left, side_right)

        # to determine how centered the car is
        small_side = abs(side_left - side_right)

        # reward if the forward distance is greater than log2(8 m)
        if forward_dist < 3.0:
            bonus -= 1.0 * (np.log2(3.0) - forward_dist)
        else:
            bonus += 1.0 * (forward_dist - np.log2(3.0))

        # reward if the side distance is greater than 2 ^ 1.5
        if side_mean < 1.5:
            bonus -= 5.0 * (np.log2(1.5) - side_mean)
        else:
            bonus += 2.0 * (side_mean - np.log2(1.5))

        # penalty for how far from the center the car is
        bonus -= 5.0 * (small_side)

        # if pressing left and car is closer to left penalize
        # if pressing right and car is closer to right penalize
        if move[2] < -0.5 and side_right < side_left:
            bonus -= 3.0 * (side_right - side_left)
        elif move[2] > 0.5 and side_right > side_left:
            bonus -= 3.0 * (side_right - side_left)

    return bonus
        

# Computes discounted future rewards and normalizes advantages.
def compute_advantage(rewards, values, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        returns[t] = running_add
    returns = torch.tensor(returns).to(device)
    advantages = returns - values.detach()

    # might consider normalizing these values
    advantages = advantages / 5.0
    return advantages, returns

# Trains the Gaussian policy agent using previously logged data.
def train(model, optimizer):
    BATCH_SIZE = 128

    # trains a small batch of the overall dataset
    def train_batch(dataset):
        model.train()
        loaders = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
        total_loss = 0.0
        for images,tel, move,  rewards in loaders:
            loss = train_step(model, optimizer, images, tel, move, rewards)
            total_loss += loss

    for ep in range(1):
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
        
    torch.save(model.state_dict(), 'test.pt')
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

    while (time.time() - start_time) < 240 and (time.time() - cp_time) < 60 and stuck_count < 60 and end_count < 15:

        # waits for the map to actually start before giving or storing commands
        if time.time() - start_time < 2:
            cp_time = time.time() + 2
            time.sleep(2)

        with state_lock:
            state = latest_state.copy()
            stacked = np.stack([np.log2(1 + f) for f in frame_buffer], axis=0)

        if state['ts'] == prev_state['ts']:
            time.sleep(1.0/60.0)
            end_count += 1
            continue
        else:
            end_count = 0
        
        # have not implemented this on accident will get back to this
        state_vec = np.array([
            [state['speed'] / 100.0,
             state['x'] / 1000.0,
             state['y'] / 100.0, 
             state['z'] / 1000.0, 
             state['cp'] / 100.0
             ]])
        
        stacked = np.expand_dims(stacked, 0)

        if abs(state['x'] - prev_state['x']) < 0.03 and abs(state['z'] - prev_state['z']) < 0.03:
            stuck_count += 1
        else:
            stuck_count = 0

        # runs the information into the model to get outputs
        with torch.no_grad():
            mu, log_std, _ = model(torch.from_numpy(stacked).float().to(device), torch.tensor(state_vec, dtype=torch.float32).to(device))
            std = torch.exp(log_std).clamp(min=1e-6)
            dist = Normal(mu, std)
  

        epsilon = 0.1
        # 10% chance to choose a random move so that the game sees more moves
        if np.random.random() < epsilon:
            action = np.random.uniform(-1.0, 1.0, size=(3,))
        else:
            action = dist.rsample()
            action = torch.tanh(action).cpu().numpy()[0]

        reward = compute_reward(prev_state, state, action, cp_time)

        bonus = lidar_clearance_bonus(state['speed'] - prev_state['speed'], action)

        reward += bonus

        forward = action[0] > 0
        back = action[1] < 0
        right = action[2] > 0.5
        left = action[2] < -0.5
        key = ((forward, back, right, left))

        action_stats[key]['count'] += 1
        action_stats[key]['reward'] += reward

        prev_state = state.copy()
        transition = {
            'state': state, # had this as state_vec[0] when I was not using else where which makes no sense to do
            'image': stacked,
            'move': action,
            'reward': reward,
            'cp': state['cp']
        }

        transitions_since_last_cp.append(transition)

        # if a new cp is reached give a reward to the states that made it get to the new cp divided by how many states
        if state['cp'] > cp_num:
            cp_reward = min(100.0 / len(transitions_since_last_cp), 5.0)
            total_weight = sum(max(0.1, t['state'][0]) for t in transitions_since_last_cp)
            avg_reward = np.mean([t['reward'] for t in transitions_since_last_cp])
            for t in transitions_since_last_cp:
                weight = max(0.1, t['state'][0]) / total_weight
                direction = np.sign(t['reward'] - avg_reward)
                scale = 1.0 + direction * 0.5
                t['reward'] += weight * cp_reward * scale
            episode_data.extend(transitions_since_last_cp)
            transitions_since_last_cp = []
            cp_num = state['cp']
            cp_time = time.time()
            
        # completes the action determined above
        if action[0] > 0:
            keyboard.press('w')
        else:
            keyboard.release('w')
        if action[1] > 0:
            keyboard.press('s')
        else:
            keyboard.release('s')
        if action[2] < -0.5:
            keyboard.press('a')
        else:
            keyboard.release('a')
        if action[2] > 0.5:
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

    # give a large reward to all of the states in the episode as they all helped get to the end
    if end_count >= 15:
        for reward in episode_data:
            reward['reward'] += 1000.0 / len(episode_data) if episode_data else 1.
    
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

    model = LSTMAgent().to(device)
    model.init_weights()
    optimizer = optim.Adam([{'params':model.mu.parameters()},{'params':model.log_std.parameters()}, {'params':model.critic.parameters(), 'lr': 5e-5}], lr=1e-5)

    read = Thread(target=read_game_state)
    read.start()

    for _ in range(BUFFER_SIZE):
        frame_buffer.append(lidar_numbers(get_screenshot()))

    cap_thread = Thread(target=capture_loop)
    cap_thread.start()
    connection_event.clear()
    connection_event.wait()
    
    for i in range(1000):
        print(i)
        action_stats = defaultdict(lambda: {'count': 0, 'reward': 0.0})
        shutdown_event.set()
        inf = Thread(target=inference, args=(model, action_stats))
        inf.start()
        inf.join()
        for (forward, back, right, left), stats in action_stats.items():
            print(f"Action ({forward}, {back}, {right}, {left}): Count = {stats['count']}, Total Reward = {stats['reward']/stats['count']:.2f}")
        shutdown_event.clear()
        model = train(model, optimizer).to(device)
        
    end_event.set()
    stop_capture.set()
    cap_thread.join()
    camera.stop()
    read.join()

if __name__ == "__main__":
    main()