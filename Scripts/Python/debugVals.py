from pynput.keyboard import Key, Listener,  Controller
import time
import os
import math
import socket
from threading import Thread
import threading
import numpy as np
import csv
import pickle
import dxcam, cv2
from collections import deque
import matplotlib.pyplot as plt


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

action_state = {'move': 2, 'turn': 2}

def on_press(key):
    try:
        c = key.char
    except AttributeError:
        return
    if c == 'w': action_state['move'] = 0
    elif c == 's': action_state['move'] = 1
    if c == 'a': action_state['turn'] = 0
    elif c == 'd': action_state['turn'] = 1

def on_release(key):
    try:
        c = key.char
    except AttributeError:
        return
    if c in ['w', 's']:
        action_state['move'] = 2
    if c in ['a', 'd']:
        action_state['turn'] = 2
    if key == Key.esc:
        return False
    
listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

def heading_reward(current_state, prev_state, cp_positions, k_heading=2.0):
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
    return - k_heading * abs(err)


def proximity_reward(current_state, prev_state, cp_positions, k_prox=3.0):
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


def compute_reward(prev_state, current_state, move, alpha=1.0, gamma=5.0):
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

def inference():
    global episode
    episode = 0
    # plt.ion()
    # fig, axes = plt.subplots(2,5, figsize=(20,10))
    cp_positions = {}
    # axes = axes.flatten()
    count = 0
    for i in range(5):
        filename = f'{i}data_log.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['prevtime', 'prevspeed', 'prevx', 'prevy', 'prevz', 'prevcp', 'currtime', 'currspeed', 'curx', 'cury', 'curz', 'curcp', 'move', 'turn', 'angle', 'dist', 'reward'])
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
        while (time.time() - start_time) < 60 and (time.time() - cp_time) < 10:
            if time.time() - start_time < 2:
                cp_time = time.time() + 2
                time.sleep(2)
            # start = time.perf_counter()
            with state_lock:
                state = latest_state.copy()
                #stacked = np.stack([f[0] for f in frame_buffer], axis=0)
            # for idx in range(BUFFER_SIZE):
            #     axes[idx].imshow(frame_buffer[idx][0], cmap='gray')
            #     axes[idx].axis('off')
            #     axes[idx].set_title(f"t-{BUFFER_SIZE - idx}")
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
            # state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['cp']]])
            # stacked = np.expand_dims(stacked, 1)
            # stacked = np.expand_dims(stacked, 0)
            move_action = action_state['move']
            turn_action = action_state['turn']
            # print('curr state ', state, ' prev state ', prev_state)
            # print('move ', move_action, ' turn ', turn_action)
            hr = heading_reward(state, prev_state, cp_positions)
            # print('err ',hr)
            dr = proximity_reward(state, prev_state, cp_positions)
            # print('prox ', dr)
            reward = compute_reward(prev_state, state, move_action)
            # print('reward ',reward, 'state ', state)
            reward += hr
            reward += dr
            if count >= 30:
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        prev_state['ts'],
                        prev_state['speed'],
                        prev_state['x'],
                        prev_state['y'],
                        prev_state['z'],
                        prev_state['cp'],
                        state['ts'],
                        state['speed'],
                        state['x'],
                        state['y'],
                        state['z'],
                        state['cp'],
                        move_action,
                        turn_action,
                        hr,
                        dr,
                        reward
                    ])
                count = 0
            count += 1
            prev_state = state.copy()
            # episode_data.append({'state':state_vec[0], 'image': stacked,'move': move_action, 'turn': turn_action,'reward': reward})
            # plt.pause(0.001)
            time.sleep(1.0/60.0)
            # for ax in axes:
            #     ax.clear()
            
            # if plt.get_fignums() == []:
            #     break

    
def main():
    init_camera()
    read = Thread(target=read_game_state)
    read.start()
    connection_event.clear()
    connection_event.wait()
    for _ in range(BUFFER_SIZE):
        frame_buffer.append(get_screenshot())
    cap_thread = Thread(target=capture_loop)
    cap_thread.start()
    # model = train(model, optimizer, cp_positions, True).to(device)
    for i in range(1):
        print(i)
        shutdown_event.set()
        inference()
        shutdown_event.clear()
        # model = train(model, optimizer, cp_positions).to(device)
    end_event.set()
    stop_capture.set()
    cap_thread.join()
    camera.stop()
    read.join()

if __name__ == "__main__":
    main()