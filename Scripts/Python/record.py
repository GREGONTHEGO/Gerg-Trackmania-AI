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
def genModel():
    inp_image = tf.keras.Input(shape=(100, 200, 3*k))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inp_image)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    img_feat = tf.keras.layers.GlobalAveragePooling2D()(x)

    #normalizer = tf.keras.layers.Normalization(axis=-1)
    inp_telemetry = tf.keras.Input(shape=(5,))
    telem_feat = tf.keras.layers.Dense(64, activation='relu')(inp_telemetry)
    merge = tf.keras.layers.Concatenate()([img_feat, telem_feat])
    #x = normalizer(inp_telemetry)
    x = tf.keras.layers.Dense(1024)(merge)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    move = tf.keras.layers.Dense(3, name='move')(x) # 0 = forward, 1 = backward, 2 = nothing
    lat = tf.keras.layers.Dense(3, name='turn')(x) # 0 = left, 1 = right, 2 = nothing
    model = tf.keras.Model(inputs=[inp_image, inp_telemetry], outputs=[move, lat])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def scale_state(state):
    state['speed'] = (state['speed']) / 100.0
    state['x'] = (state['x']) / 1000.0
    state['y'] = (state['y']) / 100.0
    state['z'] = (state['z']) / 1000.0
    state['cp'] = int(state['cp'])/ 5.0
    return state

def inference():
    global episode
    episode = 0

    start_time = time.time()
    episode_data = []
    with state_lock:
        prev_state = scale_state(latest_state)
    cp_time = time.time()
    cp_num = 0
    while (time.time() - start_time) <20 and (time.time() - cp_time) < 60:
        # if time.time() - start_time < 2:
        #     cp_time = time.time() + 2
        #     time.sleep(2)
        
        if latest_state['ts'] < prev_state['ts']:
            continue
        with state_lock:
            state = scale_state(latest_state).copy()
            image = get_screenshot()
        if state['cp'] == 5:
            break
        if state['cp'] > cp_num:
            cp_num = state['cp']
            cp_time = time.time()
        state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['cp']]])
        #print(state_vec)
        frame_buffer.append(image)
        stacked = np.stack([f[0] for f in frame_buffer], axis=0)
        stacked = np.expand_dims(stacked, 0)
        move = action_state['move']
        turn = action_state['turn']
        reward = compute_reward(prev_state, state)
        prev_state = state.copy()
        episode_data.append({'state':state_vec[0], 'image': stacked,'move': move, 'turn': turn,'reward': reward})
        time.sleep(1.0/20.0)
    with open(f'{episode}turn.pkl', 'wb') as f:
        pickle.dump(episode_data, f)

    

if __name__ == "__main__":
    # model = genModel()
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
    # model.predict([stacked ,dummy])   
    shutdown_event.clear()
    inf = Thread(target=inference)
    inf.start()
    inf.join()
    end_event.set()
    camera.stop()
    read.join()