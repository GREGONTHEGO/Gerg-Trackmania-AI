from pynput.keyboard import Key, Controller
import time
import math
import socket
import sys
import threading
from threading import Thread
from collections import deque

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SEED = 5
np.random.seed(SEED)
torch.manual_seed(SEED)

if sys.platform == "win32":
    import dxcam
else:
    import mss



# Visual input / replay efficiency
BUFFER_SIZE = 5                  
CAPTURE_FPS = 60.0
RESIZED_W, RESIZED_H = 160, 90   

# Screen capture region (left, top, right, bottom)
WINDOW_REGION = (640, 300, 1920, 1200)

HOST = '127.0.0.1'
PORT = 5055

# Episode controls
MAX_IDLE_SECONDS = 30            
MAX_STUCK_STEPS = 200            
SLEEP_PER_STEP = 1.0 / 60.0

# distance from line before not getting reward
OFFTRACK_MAX_DIST = 40.0   

# Runs two forward biased runs that will get as far
# into the map as possible so the agent has a good guide
DEMO_BIAS_EPISODES = 2      
EP_TRAIN_START = 2          

# Ghost smoothing
GHOST_N = 10000            
SMOOTH_ITERS = 1     

# Path Following reward scale
IDX_REWARD_DIVISOR = 2.0
PROGRESS_SCALE = 5.0

# Trajectory Parameters
NB_OBS_FORWARD = 250           
NB_OBS_BACKWARD = 250          
NB_ZERO_REW_BEFORE_FAILURE = 120 
MAX_DIST_FROM_TRAJ_TMRL = OFFTRACK_MAX_DIST

# Fallback (Speed) & Penalties
# should not run unless there are parts of the map
# that the AI doesnt get to
FALLBACK_SCALE = 1.0           
ALIGNMENT_REWARD_SCALE = 0.5 
ALIGNMENT_LOOKAHEAD = 50     
STUCK_PENALTY = -10.0
LIVING_REWARD = 0.1

# DQN config
ACTION_SIZE = 9 # [w, a, s, d, wa, wd, sa, sd, '']
STATE_SIZE = 2 # [gear, rpm]

# Teacher/CNN mixing
BIAS_EP0_FORWARD_P = 0.85   
BIAS_EP1_FORWARD_P = 0.60   
TEACHER_CNN_TOPK = 3        
TEACHER_CNN_TAU = 0.8       

CURRENT_KEYS = set()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_buffer = deque(maxlen=BUFFER_SIZE)
camera = None

state_lock = threading.Lock()
end_event = threading.Event()
connection_event = threading.Event()
stop_capture = threading.Event()

keyboard = Controller()

# stores latest state
latest_state = {
    'ts': 0.0,
    'speed': 0.0,
    'x': 0.0,
    'y': 0.0,
    'z': 0.0,
    'cp': 0.0,
    'gear': 0.0, 
    'rpm': 0.0   
}

# Best episode / path tracking
best_episode_data = None
best_cp = -1
best_reward = -float('inf')
best_path = None       
best_cp_times = {}     
best_max_cp = -1
best_total_time = float('inf')

# if is for windows else is for linux
def init_camera():
    global camera
    if sys.platform == "win32":
        camera = dxcam.create(
            output_idx=0,
            output_color="GRAY",
            region=WINDOW_REGION,
            max_buffer_len=10
        )
        camera.start(target_fps=int(CAPTURE_FPS))
    else:
        camera = mss.mss()


# gathers and cuts the screenshot down to a grey scale
# and stores in values from 0 to 255
def get_screenshot():
    if sys.platform == "win32":
        if camera is None:
            return np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
        frame = camera.get_latest_frame()
        if frame is None:
            return np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3:
            c = frame.shape[2]
            if c == 1: gray = frame[:, :, 0]
            elif c == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif c == 4: gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            else: gray = frame[:, :, 0]
        else:
            gray = np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
    else:
        left, top, right, bottom = WINDOW_REGION
        monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        img = np.array(camera.grab(monitor))
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    gray_resized = cv2.resize(gray, (RESIZED_W, RESIZED_H), interpolation=cv2.INTER_AREA)
    return gray_resized.astype(np.uint8)


# thread that gathers the screenshots
def capture_loop():
    dt = 1.0 / CAPTURE_FPS
    while not stop_capture.is_set():
        frame = get_screenshot()
        frame_buffer.append(frame)
        time.sleep(dt)


# if the buffer is empty fill with one frame and 
# if not quite full duplicate the last frame 
# then return the stack of frames
def get_stacked_frames_u8():
    if len(frame_buffer) == 0:
        frame = get_screenshot()
        for _ in range(BUFFER_SIZE):
            frame_buffer.append(frame)
    elif len(frame_buffer) < BUFFER_SIZE:
        last = frame_buffer[-1]
        for _ in range(BUFFER_SIZE - len(frame_buffer)):
            frame_buffer.append(last)
    return np.stack(frame_buffer, axis=0)


# given a line will parse and save it to its separate parts
def parse_telem(line: str):
    if not line: return
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6: return

    try:
        ts = float(parts[0])
        fwd = float(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        cp = float(parts[5])
        
        gear = 0.0
        rpm = 0.0
        
        if len(parts) >= 7 and parts[6] != "":
            try: gear = float(parts[6])
            except ValueError: gear = 0.0
            
        if len(parts) >= 8 and parts[7] != "":
            try: rpm = float(parts[7])
            except ValueError: rpm = 0.0

    except ValueError:
        return

    with state_lock:
        latest_state["ts"] = ts
        latest_state["speed"] = fwd
        latest_state["x"] = x
        latest_state["y"] = y
        latest_state["z"] = z
        latest_state["cp"] = cp
        latest_state["gear"] = gear
        latest_state["rpm"] = rpm


# grabs the latest state and locks it so that there isnt a partial update
def get_latest_state():
    with state_lock:
        return dict(latest_state)


# opens a socket to listen to the plugin that Greg wrote that 
# gathers data from the game like [ts, x, y, z, cp, gear, rpm]
def read_game_state():
    while not end_event.is_set():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, PORT))
                s.listen(1)
                print(f"[telemetry] Listening on {HOST}:{PORT}")
                conn, addr = s.accept()
                print(f"[telemetry] Connected by {addr}")
                connection_event.set()
                conn.settimeout(0.1)
                buf = ""
                try:
                    while not end_event.is_set():
                        try:
                            data = conn.recv(4096)
                            if not data: break
                            buf += data.decode("utf-8", errors="ignore")
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                parse_telem(line.strip())
                        except socket.timeout: continue
                        except Exception as e: break
                finally:
                    try: conn.close()
                    except: pass
                    print("[telemetry] Connection closed")
                    time.sleep(0.5)
        except Exception as e:
            print(f"[telemetry] Listener error: {e}")
            time.sleep(1.0)



def build_path_and_cp_times(episode_data, min_step=0.1):
    points = []
    cp_times = {}
    if not episode_data: return points, cp_times
    last_x = last_z = None
    s = 0.0
    for t in episode_data:
        x, z = t.get('x'), t.get('z')
        if x is None or z is None: continue
        cp = int(t.get('cp', 0))
        ts = t.get('ts')
        if ts is not None and cp > 0 and cp not in cp_times:
            cp_times[cp] = ts
        if last_x is not None:
            ds = math.hypot(x - last_x, z - last_z)
            if ds < min_step: continue
            s += ds
        points.append({"x": x, "z": z, "s": s, "cp": cp})
        last_x, last_z = x, z
    return points, cp_times


# smooth the corners as driving with a keyboard would add jagged noise
def corner_cut(P, iters=1):
    P = np.asarray(P, np.float32)
    if len(P) < 3 or iters <= 0: 
        return P
    for _ in range(iters):
        Q = [P[0]]
        for i in range(len(P) - 1):
            q = 0.75 * P[i] + 0.25 * P[i + 1]
            r = 0.25 * P[i] + 0.75 * P[i + 1]
            Q.extend([q, r])
        Q.append(P[-1])
        P = np.asarray(Q, np.float32)
    return P


# evenly space reward points along the ghost path
def respace_poly(P, n=GHOST_N):
    P = np.asarray(P, np.float32)
    if len(P) < 2: 
        return np.repeat(P[:1], n, axis=0) if len(P) else np.zeros((n, 2), np.float32)
    segs = P[1:] - P[:-1]
    dists = np.sqrt((segs ** 2).sum(axis=1))
    total = float(np.sum(dists))
    if total <= 1e-6: 
        return np.repeat(P[:1], n, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(dists)])
    targets = np.linspace(0.0, total, n)
    out = []
    j = 0
    for t in targets:
        while j < len(dists) and cum[j + 1] < t: 
            j += 1
        if j >= len(dists):
            out.append(P[-1])
            continue
        if dists[j] < 1e-6:
            out.append(P[j])
            continue
        ratio = (t - cum[j]) / dists[j]
        out.append(P[j] + ratio * (P[j + 1] - P[j]))
    return np.asarray(out, np.float32)


# maps checkpoint data and distance traveled to the smooth path
def rebuild_dist_and_cp(P_rs, path_orig):
    s_orig = np.array([p["s"] for p in path_orig], dtype=np.float32)
    cp_orig = np.array([int(p["cp"]) for p in path_orig], dtype=np.int32)
    segs = P_rs[1:] - P_rs[:-1]
    dists = np.sqrt((segs ** 2).sum(axis=1))
    s_new = np.concatenate([[0.0], np.cumsum(dists)])
    cp_new = []
    for s in s_new:
        j = int(np.clip(np.searchsorted(s_orig, s, side="left"), 0, len(s_orig) - 1))
        cp_new.append(int(cp_orig[j]))
    out = []
    for (x, z), s, cp in zip(P_rs, s_new, cp_new):
        out.append({"x": float(x), "z": float(z), "s": float(s), "cp": int(cp)})
    return out


# calls the smooth function corner_cut and the resample function respace_poly then
# returns the path with the checkpoint and length mapped
def smooth_and_resample_path(path, smooth_iters=SMOOTH_ITERS, n=GHOST_N):
    if not path or len(path) < 2: return path
    P = np.array([[p["x"], p["z"]] for p in path], dtype=np.float32)
    P = corner_cut(P, iters=smooth_iters) if smooth_iters > 0 else P
    P_rs = respace_poly(P, n=n)
    return rebuild_dist_and_cp(P_rs, path)


# uses vector math find where the car is based on the relative path
def closest_prog_on_path(x, z, path, max_dist=OFFTRACK_MAX_DIST):
    if not path or len(path) < 2: return None
    best_s = None
    best_d2 = None
    max_d2 = max_dist * max_dist if max_dist is not None else None
    for i in range(len(path) - 1):
        ax, az, as_ = path[i]["x"], path[i]["z"], path[i]["s"]
        bx, bz, bs_ = path[i + 1]["x"], path[i + 1]["z"], path[i + 1]["s"]
        vx, vz = bx - ax, bz - az
        wx, wz = x - ax, z - az
        seg_len2 = vx * vx + vz * vz
        if seg_len2 == 0.0: t = 0.0
        else:
            t = (wx * vx + wz * vz) / seg_len2
            t = max(0.0, min(1.0, t))
        px = ax + t * vx
        pz = az + t * vz
        d2 = (x - px) ** 2 + (z - pz) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_s = as_ + t * (bs_ - as_)
    if best_d2 is None: return None
    if max_d2 is not None and best_d2 > max_d2: return None
    return best_s


# Tracks progress along the specific path
class AIGhostTMReward:
    def __init__(self, nb_obs_forward=NB_OBS_FORWARD, nb_obs_backward=NB_OBS_BACKWARD, nb_zero_rew_before_failure=NB_ZERO_REW_BEFORE_FAILURE,
                 max_dist_from_traj=MAX_DIST_FROM_TRAJ_TMRL, idx_reward_divisor=IDX_REWARD_DIVISOR):
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.max_dist_from_traj = float(max_dist_from_traj)
        self.idx_reward_divisor = float(idx_reward_divisor)
        self.data = None
        self.datalen = 0
        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.steps_since_reset = 0

    # returns true if a valid ghost in currently loaded
    def has_traj(self):
        return self.data is not None and self.datalen > 1

    # resets the index counter to the start of the path
    def reset(self):
        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.steps_since_reset = 0

    # if car moves forward along traj give it a positive reward
    def compute_reward(self, pos_x, pos_z):
        terminated = False
        if not self.has_traj():
            return 0.0, terminated

        pos = np.array([pos_x, pos_z], dtype=np.float32)
        self.step_counter += 1
        self.steps_since_reset += 1

        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = self.cur_idx

        while True:
            if index >= self.datalen: break
            dist = np.linalg.norm(pos - self.data[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            if index >= self.datalen or temp <= 0:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx
                break

        reward = (best_index - self.cur_idx) / self.idx_reward_divisor

        if best_index == self.cur_idx:
            if self.steps_since_reset > 60: 
                min_dist = np.inf
                index = self.cur_idx
                temp = self.nb_obs_backward
                while True:
                    if index <= 0: break
                    dist = np.linalg.norm(pos - self.data[index])
                    if dist <= min_dist:
                        min_dist = dist
                        best_index = index
                        temp = self.nb_obs_backward
                    index -= 1
                    temp -= 1
                    if index <= 0 or temp <= 0: break

                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True
        else:
            self.failure_counter = 0

        self.cur_idx = best_index
        return float(reward), terminated


reward_engine = AIGhostTMReward()


# fallback reward for the section that the agent hasn't reached yet
def speed_reward(prev_state, current_state, alpha=1.0):
    speed = current_state['speed']
    prev_speed = prev_state['speed']
    reward = 0.0
    
    if speed > 1.0:
        reward += (speed * 0.1) 

    delta_spd = speed - prev_speed
    if delta_spd > 0.2: 
        reward += 2.0 * delta_spd 

    if speed < 1.0:
        reward -= 0.5
        
    return reward


# checks if a ghost path exists 
def compute_reward(prev_state, current_state, action: int):
    total_reward = 0.0

    if reward_engine.has_traj():
        r, term, dbg = reward_engine.compute_reward(
            pos_x=float(current_state['x']),
            pos_z=float(current_state['z'])
        )
        
        total_reward += r * PROGRESS_SCALE
        total_reward += LIVING_REWARD
        return total_reward, term, "path_follow", dbg

    if best_path and len(best_path) > 1:
        s_proj = closest_prog_on_path(
            float(current_state['x']),
            float(current_state['z']),
            best_path,
            max_dist=OFFTRACK_MAX_DIST
        )
        if s_proj is None:
            r = speed_reward(prev_state, current_state)
            total_reward += FALLBACK_SCALE * r
            total_reward += LIVING_REWARD
            return total_reward, False, "fallback_speed"

    r = speed_reward(prev_state, current_state)
    total_reward += FALLBACK_SCALE * r 
    total_reward += LIVING_REWARD
    return total_reward, False, "pure_speed"


# make a CNN that combines with 2 telemetry points and outputs an action
class Conv2DAgent(nn.Module):
    def __init__(self, num_actions, in_channels=BUFFER_SIZE, telem_dim=STATE_SIZE):
        super().__init__()
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, RESIZED_H, RESIZED_W)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            vis_dim = x.view(1, -1).shape[1]

        self.telem_norm = nn.LayerNorm(telem_dim)
        self.telemetry_mlp = nn.Sequential(
            nn.Linear(telem_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(vis_dim + 64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, num_actions)

        self._init_weights()

    # adds a forward bias such that the first few runs do well before it 
    # starts learning
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)

    # sends info through the model
    def forward(self, x, telem):
        B = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(B, -1)
        t = self.telem_norm(telem)
        t = self.telemetry_mlp(t)
        m = torch.cat([x, t], dim=1)
        x = F.relu(self.fc1(m))
        x = F.relu(self.fc2(x))
        q_values = self.head(x)
        return q_values


# Stores a bunch of information about every step
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # pushes to the buffer
    def push(self, state, frames_u8, action, reward, next_frames_u8, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state.astype(np.float32),
            frames_u8.astype(np.uint8),
            int(action),
            float(reward),
            next_frames_u8.astype(np.uint8),
            next_state.astype(np.float32),
            float(done),
        )
        self.position = (self.position + 1) % self.capacity

    # grabs a random sample of size batch_size
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, images, actions, rewards, next_images, next_states, dones = [], [], [], [], [], [], []
        for i in idxs:
            s, img, a, r, nimg, ns, d = self.buffer[i]
            states.append(s)
            images.append(img)
            actions.append(a)
            rewards.append(r)
            next_images.append(nimg)
            next_states.append(ns)
            dones.append(d)
        states = np.vstack(states).astype(np.float32)
        images = np.stack(images).astype(np.float32) / 255.0
        next_images = np.stack(next_images).astype(np.float32) / 255.0
        next_states = np.vstack(next_states).astype(np.float32)
        actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        dones = np.array(dones, dtype=np.float32).reshape(-1, 1)
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(images).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(rewards).to(device),
            torch.from_numpy(next_images).to(device),
            torch.from_numpy(next_states).to(device),
            torch.from_numpy(dones).to(device),
        )


    def __len__(self):
        return len(self.buffer)

# the agent that trains the network of two conv2d models
class DDQNAgent:
    # initializes all of the variables and nets
    def __init__(self, state_size, action_size, learning_rate,
                 capacity, discount_factor, tau, update_every, batch_size):
        self.learn_warmup_steps = 2500 
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.local_net = Conv2DAgent(action_size, telem_dim=state_size).to(device)
        self.target_net = Conv2DAgent(action_size, telem_dim=state_size).to(device)
        self.optimizer = optim.Adam(self.local_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity)
        self.steps = 0
        self.hard_update(self.target_net, self.local_net)

    # decides on a move using Epsilon-Greedy
    def act(self, frames_u8, state_vec, eps=0.1):
        self.local_net.eval()
        with torch.no_grad():
            img = torch.from_numpy(frames_u8.astype(np.float32) / 255.0).unsqueeze(0).to(device)
            state = torch.from_numpy(state_vec.astype(np.float32)).to(device)
            q_values = self.local_net(img, state)
        self.local_net.train()
        if np.random.rand() > eps:
            return int(torch.argmax(q_values, dim=1).item())
        else:
            return int(np.random.randint(self.action_size))

    # every frame save to the buffer if buffer is full then learn
    def step(self, state, frames_u8, action, reward, next_frames_u8, next_state, done):
        self.replay_buffer.push(state, frames_u8, action, reward, next_frames_u8, next_state, done)
        self.steps += 1
        if (self.steps >= self.learn_warmup_steps and
                self.steps % self.update_every == 0 and
                len(self.replay_buffer) >= self.batch_size):
            experiences = self.replay_buffer.sample(self.batch_size)
            self.learn(experiences)

    # performs the math to update the nn using loss based on the actions and rewards
    def learn(self, experiences):
        states, images, actions, rewards, next_images, next_states, dones = experiences
        with torch.no_grad():
            next_q_local = self.local_net(next_images, next_states)
            best_next_actions = next_q_local.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_images, next_states)
            q_targets_next = next_q_target.gather(1, best_next_actions)
            q_targets = rewards + self.gamma * q_targets_next * (1.0 - dones)
        q_expected = self.local_net(images, states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_net, self.target_net)
        
    # used to make the inital weights the same
    @staticmethod
    def hard_update(target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    # slowly blends the weights of the local network with the target to keep training stable
    def soft_update(self, local_model, target_model):
        for t, s in zip(target_model.parameters(), local_model.parameters()):
            t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)



# returns a set of actions
def action_to_keys(action: int):
    if action == 0: return {"w"}
    elif action == 1: return {"w", "a"}
    elif action == 2: return {"w", "d"}
    elif action == 3: return {"s"}
    elif action == 4: return {"s", "a"}
    elif action == 5: return {"s", "d"}
    elif action == 6: return set()
    elif action == 7: return {"a"}
    elif action == 8: return {"d"}
    else: return set()


# press actions and will keep current actions pressed and only release if it is not part of the
# new action so if w,a is pressed then w we would only release a
def apply_action(action: int):
    global CURRENT_KEYS
    desired = action_to_keys(action)
    to_release = CURRENT_KEYS - desired
    to_press = desired - CURRENT_KEYS
    for k in to_release: keyboard.release(k)
    for k in to_press: keyboard.press(k)
    CURRENT_KEYS = desired


# releases all keys
def release_all_keys():
    global CURRENT_KEYS
    for k in list(CURRENT_KEYS): keyboard.release(k)
    CURRENT_KEYS.clear()


# presses delete and enter as both are used to reset runs depending on if stuck or make it to end
def reset_run():
    release_all_keys()
    keyboard.press(Key.delete)
    keyboard.release(Key.delete)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    time.sleep(0.5)


# the main loop for one attempt
def run_episode(agent, epsilon, episode_index: int):
    reset_run()
    reward_engine.reset()
    # finds location along the path
    if best_path and len(best_path) > 1 and reward_engine.has_traj():
        with state_lock: _st = latest_state.copy()
        s0 = closest_prog_on_path(float(_st['x']), float(_st['z']), best_path, max_dist=OFFTRACK_MAX_DIST)
        if s0 is not None:
            s_array = np.array([p["s"] for p in best_path], dtype=np.float32)
            reward_engine.cur_idx = int(np.clip(np.searchsorted(s_array, s0, side="left"), 0, len(s_array)-1))

    is_demo = (episode_index < DEMO_BIAS_EPISODES)
    start_time = time.time()
    last_progress_time = start_time
    episode_data = []

    with state_lock: state = latest_state.copy()
    frames = get_stacked_frames_u8()
    
   
    state_vec = np.array([[
        state['gear'] / 5.0,
        state['rpm'] / 10000.0
    ]], dtype=np.float32)

    if is_demo:
        bias_prob = BIAS_EP0_FORWARD_P if episode_index == 0 else BIAS_EP1_FORWARD_P
        if np.random.rand() < bias_prob:
            action = 0
        else:
            action = np.random.randint(ACTION_SIZE)
    else:
        action = agent.act(frames, state_vec, epsilon)
    apply_action(action)

    cp_num = int(state.get('cp', 0))
    max_cp_seen = cp_num
    stuck_steps = 0
    ep_return = 0.0
    no_ts_change_steps = 0

    while True:
        now = time.time()
        if now - last_progress_time > MAX_IDLE_SECONDS: break
        time.sleep(SLEEP_PER_STEP)

        with state_lock: next_state = latest_state.copy()

        # timestep only changes during the race at the end it will
        # send the same state repeatedly
        if next_state['ts'] == state['ts']:
            no_ts_change_steps += 1
            if no_ts_change_steps > 10.0:
                print("[episode] Map Completed, ending episode.")
                break
            continue
        else:
            no_ts_change_steps = 0

        next_frames = get_stacked_frames_u8()

        next_state_vec = np.array([[
            next_state['gear'] / 5.0,
            next_state['rpm'] / 10000.0
        ]], dtype=np.float32)

        dx = next_state['x'] - state['x']
        dz = next_state['z'] - state['z']
        dist = math.hypot(dx, dz)
        

        if next_state['speed'] < 1.0 and dist < 0.05:
            stuck_steps += 1
        else:
            stuck_steps = 0
            if is_demo: last_progress_time = now

        reward, terminated, reward_type = compute_reward(state, next_state, action)

        if stuck_steps > MAX_STUCK_STEPS:
            reward = STUCK_PENALTY
            terminated = True
            reward_type = "stuck_timeout"
            print(f"[episode] Terminated: Stuck for {MAX_STUCK_STEPS} frames.")


        # the goal of the demo is to get a run that gets as far into the
        # map as possible and then learn off a run that relatively follows
        # then go to more random learning after those to train on
        if is_demo:
            bias_prob = BIAS_EP0_FORWARD_P if episode_index == 0 else BIAS_EP1_FORWARD_P
            if np.random.rand() < bias_prob:
                next_action = 0
            else:
                next_action = np.random.randint(ACTION_SIZE)
        else:
            if (reward > 0 and reward_type == "path_follow") or (next_state['cp'] > cp_num):
                last_progress_time = now
            next_action = agent.act(next_frames, next_state_vec, epsilon)

        episode_data.append({
            'state': state_vec[0].copy(),
            'image': frames.copy(),
            'action': int(action),
            'reward': float(reward), 
            'reward_type': reward_type, 
            'cp': float(state['cp']),
            'ts': int(state['ts']),
            'x': float(state['x']),
            'y': float(state['y']),
            'z': float(state['z']),
        })

        if episode_index > 0:
            done_flag = 1.0 if terminated else 0.0
            agent.step(
                state_vec[0].copy(), frames.copy(), int(action), float(reward),
                next_frames.copy(), next_state_vec[0].copy(), float(done_flag)
            )

        if not is_demo:
            ep_return += reward
            if terminated:
                break 
        
        if is_demo and terminated:
             print("[episode] Demo run stuck, resetting...")
             break

        if next_state['cp'] > cp_num:
            cp_num = int(next_state['cp'])
        if cp_num > max_cp_seen:
            max_cp_seen = cp_num

        action = next_action
        apply_action(action)
        state = next_state
        state_vec = next_state_vec
        frames = next_frames

    release_all_keys()

    return episode_data, max_cp_seen, ep_return


# check if the most recent run is better than the last best. if so will process the path and
# set it as the new ghost
def update_best_episode(episode_data, cp_num, cur_reward, episode_index: int):
    global best_episode_data, best_cp, best_reward
    global best_path, best_cp_times, best_max_cp, best_total_time

    if not episode_data or not episode_data[0].get('ts'): 
        return
    
    path, cp_times = build_path_and_cp_times(episode_data)
    ts0 = episode_data[0].get('ts') 
    
    if cp_times and ts0 is not None:
        max_cp = max(cp_times.keys())
        lap_time = (cp_times[max_cp] - ts0) / 1000.0 
    else:
        max_cp = int(cp_num)
        lap_time = float('inf')

    improve = False
    if max_cp > best_max_cp: improve = True
    elif max_cp == best_max_cp and lap_time < best_total_time: improve = True

    path_sr = smooth_and_resample_path(path, smooth_iters=SMOOTH_ITERS, n=GHOST_N)

    if improve:
        best_episode_data = list(episode_data)
        best_cp = max_cp
        best_reward = cur_reward
        best_path = path_sr 
        best_cp_times = cp_times
        best_max_cp = max_cp
        best_total_time = lap_time
        print(f"[best_path] New best: CP {best_max_cp}, time {best_total_time:.3f}s, len {len(best_path)}")


# initializes all of the different threads and loops through 5000 episodes while slowly decreasing
# epsilon
def main():
    init_camera()
    agent = DDQNAgent(
        state_size=STATE_SIZE, action_size=ACTION_SIZE,
        learning_rate=1e-4, capacity=5000, discount_factor=0.99,
        tau=1e-3, update_every=8, batch_size=64,
    )

    read_thread = Thread(target=read_game_state, daemon=True)
    read_thread.start()
    connection_event.wait()
    for _ in range(BUFFER_SIZE):
        frame_buffer.append(get_screenshot())
    cap_thread = Thread(target=capture_loop, daemon=True)
    cap_thread.start()

    epsilon = 0.25
    for ep in range(5001): 
        has_best = best_path is not None and len(best_path) > 0
        print(f"\n[episode] {ep} | eps={epsilon:.3f} | demo={(ep < DEMO_BIAS_EPISODES)} | has_best={has_best}")
        episode_data, cp_num, ep_ret = run_episode(agent, epsilon, ep)
        
        update_best_episode(episode_data, cp_num, ep_ret, ep) 
        
        epsilon = max(epsilon - 0.001, 0.05)

    end_event.set()
    stop_capture.set()
    release_all_keys()
    if sys.platform == "win32" and camera is not None:
        try: camera.stop()
        except: pass
    print("[main] Training loop finished.")



if __name__ == "__main__":
    main()