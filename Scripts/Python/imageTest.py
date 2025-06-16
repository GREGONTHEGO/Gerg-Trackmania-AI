from pynput.keyboard import Key, Listener, Controller
import time
import os
import socket
from threading import Thread
import csv
import threading
import tensorflow as tf
import numpy as np
import mss
import pickle
import dxcam, cv2
from collections import deque
import matplotlib.pyplot as plt

k = 15
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

def compute_reward(prev_state, current_state, alpha=1.0, gamma=5.0):
    speed = current_state['speed'] * 100
    cp_diff = (current_state['cp'] - prev_state['cp']) * 5
    cp_reward = gamma * max(cp_diff, 0)

    return alpha * speed + cp_reward


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
        prev_time = 0
        while not end_event.is_set():
            try:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break
                parts = data.split('\n')
                #print(parts)
                if parts[-1] == '':
                    last_line = parts[-2]
                else:
                    last_line = parts[-1]
                fields = last_line.split(',')
                if len(fields) == 6 and all(fields):
                    ts, forwardVel, x, y, z, cp = fields
                    #print(fields)
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

def scale_state(state):
    state['speed'] = (state['speed']) / 100.0
    state['x'] = (state['x']) / 1000.0
    state['y'] = (state['y']) / 100.0
    state['z'] = (state['z']) / 1000.0
    state['cp'] = int(state['cp'])/ 5.0
    return state

def inference():
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(3, 5, figsize=(12, 6))
    axes = axes.flatten()

    global episode
    episode = 0

    with state_lock:
        prev_state = scale_state(latest_state)
    frame_buffer.extend([get_screenshot()] * k)

    while True:
        with state_lock:
            state = scale_state(latest_state).copy()
            image = get_screenshot()
        frame_buffer.append(image)

        # Plot each frame
        for idx in range(k):
            axes[idx].imshow(frame_buffer[idx][0], cmap='gray')
            axes[idx].axis('off')
            axes[idx].set_title(f"t-{k - idx}")
        
        plt.pause(0.001)  # Allow update
        for ax in axes:
            ax.clear()

        if plt.get_fignums() == []:
            break  # Window closed by user

        time.sleep(0.05)

    

if __name__ == "__main__":
    read = Thread(target=read_game_state)
    read.start()
    connection_event.clear()
    connection_event.wait()
    dummy = np.zeros((1, 5), dtype=np.float32)
    image = get_screenshot()
    frame_buffer.extend([image] * k)
    print(len(frame_buffer))
    print(f[0].shape for f in frame_buffer)
    stacked = np.stack([f[0] for f in frame_buffer], axis=0)
    stacked = np.expand_dims(stacked, 0)
    print(stacked.shape) 
    shutdown_event.clear()
    inf = Thread(target=inference)
    inf.start()
    inf.join()
    end_event.set()
    camera.stop()
    read.join()