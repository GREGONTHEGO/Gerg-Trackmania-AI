from pynput.keyboard import Key, Listener, Controller
import time
import socket
from threading import Thread
import threading
import numpy as np
import pickle
import dxcam, cv2
from collections import deque
import os, math

k = 10
frame_buffer = deque(maxlen=k)

region = (640, 300, 1920, 1200)
camera = dxcam.create(output_idx=0, output_color='GRAY', region=region)

camera.start()
state_lock = threading.Lock()
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

def get_screenshot():
    screenshot = camera.get_latest_frame()
    img = screenshot[:, :, :3]  
    img = cv2.resize(img, (200, 100))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

HOST = '127.0.0.1'
PORT = 5055
keyboard = Controller()
shutdown_event = threading.Event()
end_event = threading.Event()
connection_event = threading.Event()

latest_state = {'ts':0,'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'cp': 0}
episode = 0

def heading_reward(current_state, prev_state, cp_positions, k_heading=1.0):
    dx = current_state['x'] - prev_state['x']
    dz = current_state['z'] - prev_state['z']
    if dx==0 and dz==0:
        return 0.0
    actual_heading = math.atan2(dz, dx)

    next_cp = int(current_state['cp']) + 1
    if next_cp in cp_positions:
        x_cp, z_cp = cp_positions[next_cp]['x'], cp_positions[next_cp]['z']
        desired_heading = math.atan2(z_cp - current_state['z'], x_cp - current_state['z'])
    else:
        desired_heading = actual_heading

    err = (actual_heading - desired_heading + math.pi) % (2*math.pi) - math.pi
    return - k_heading * abs(err)

def compute_reward(prev_state, current_state, move, alpha=1.0, gamma=1.0):
    speed = current_state['speed']
    prev_speed = prev_state['speed']
    cp_diff = (current_state['cp'] - prev_state['cp'])
    cp_reward = gamma * max(cp_diff, 0)

    reward = alpha * max(0.0,speed) + cp_reward
    if not move == 2 and abs(current_state['x'] - prev_state['x']) < 0.1 and abs(current_state['z'] - prev_state['z']) < 0.1:
        return -5.0
    # if speed < 0.0:
    #     return -2.0
    delta_spd = speed - prev_speed
    if (delta_spd) > 5.0 and speed > 0.0:
        reward -= 10.0 * (delta_spd - 5.0)
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
                        latest_state['cp'] = int(cp)
                        #latest_state['pic'] = get_screenshot()
            except Exception as e:
                print("Error reading data:", e)
                break
        conn.close()

def inference():
    global episode
    episode = 0
    CP_POSITIONS_FILE = 'cp_positions.pkl'
    if os.path.exists(CP_POSITIONS_FILE):
        with open(CP_POSITIONS_FILE, 'rb') as f:
            cp_positions = pickle.load(f)
    else:
        cp_positions = {}
    start_time = time.time()
    episode_data = []
    with state_lock:
        prev_state = latest_state
    cp_time = time.time()
    cp_num = 0
    while (time.time() - start_time) <15 and (time.time() - cp_time) < 60:
        if latest_state['ts'] < prev_state['ts']:
            continue
        with state_lock:
            state = latest_state.copy()
            image = get_screenshot()
        if state['cp'] > cp_num:
            if state['cp'] in cp_positions:
                cur_time = time.time() - start_time
                if cp_positions[state['cp']]['time'] > cur_time:
                    cp_positions[state['cp']] = {'time': cur_time, 'x': state['x'], 'z': state['z']}
            else:
                cur_time = time.time() - start_time
                cp_positions[state['cp']] = {'time': cur_time, 'x': state['x'], 'z': state['z']}
            cp_num = state['cp']
        state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['cp']]])
        frame_buffer.append(image)
        stacked = np.stack([f[0] for f in frame_buffer], axis=0)
        stacked = np.expand_dims(stacked, 1)
        stacked = np.expand_dims(stacked, 0)
        move = action_state['move']
        turn = action_state['turn']
        hr = heading_reward(state, prev_state, cp_positions)
        print('err ',hr)
        reward = compute_reward(prev_state, state, move)
        print('reward ',reward, 'state ', state)
        reward += hr
        if move != 2:
            reward += 0.2
        prev_state = state.copy()
        episode_data.append({'state':state_vec[0], 'image': stacked,'move': move, 'turn': turn,'reward': reward})
        time.sleep(1.0/60.0)
    with open(f'{episode}turn.pkl', 'wb') as f:
        pickle.dump(episode_data, f)

    

if __name__ == "__main__":
    read = Thread(target=read_game_state)
    read.start()
    connection_event.clear()
    connection_event.wait()
    image = get_screenshot()
    frame_buffer.extend([image] * k)
    shutdown_event.clear()
    inf = Thread(target=inference)
    inf.start()
    inf.join()
    end_event.set()
    camera.stop()
    read.join()