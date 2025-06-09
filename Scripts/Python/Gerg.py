from pynput.keyboard import Key, Controller
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

def get_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = img[:, :, :3]  
        img = tf.image.resize(img, (200, 100))
        return img.numpy()

HOST = '127.0.0.1'
PORT = 5055
keyboard = Controller()
shutdown_event = threading.Event()
connection_event = threading.Event()

latest_state = {'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'cp': 0}
episode = 0

def compute_reward(prev_state, current_state, alpha=1.0, beta=1.0, gamma=100.0):
    speed = current_state['vx']
    delta_x = current_state['x'] - prev_state['x']
    delta_cp = current_state['cp'] - prev_state['cp']
    print("reward:", alpha * speed + beta * delta_x + gamma * delta_cp)
    return alpha * speed + beta * delta_x + gamma * delta_cp

def policy_loss(logits, actions, rewards):
    logp = actions * tf.math.log(logits + 1e-10) + (1 - actions) * tf.math.log(1 - logits + 1e-10)
    return -tf.reduce_mean(tf.reduce_sum(logp, axis=1) * rewards)

def train(model, normalizer, episodes=50):
    for ep in range(episode):
        print(ep)
        with open(f'{ep}statesInEpoch.pkl', 'rb') as f:
            statesInEpoch = pickle.load(f)
        states = np.stack([state['state'] for state in statesInEpoch], axis=0).astype(np.float32)
        actions = np.stack([state['action'] for state in statesInEpoch], axis=0).astype(np.float32)
        rewards = np.array([state['reward'] for state in statesInEpoch], dtype=np.float32)
        normalizer.adapt(states)
        with tf.GradientTape() as tape:
            logits = model(states)
            logits = tf.nn.sigmoid(logits)
            loss = policy_loss(logits, actions, rewards)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Episode {episode}, Loss: {loss.numpy()}")
    
        # Train the model using the statesInEpoch
        # Do not execute keypresses here, just train the model
        
    model.save('model_weights.keras', save_format='keras')
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
        while not shutdown_event.is_set():
            try:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break
                #print("received data:", data)
                for line in data.strip().splitlines():
                    if line.startswith("Speed="):
                        latest_state['speed'] = float(line.split('=')[1])
                    elif line.startswith("Position="):
                        parts = line.replace("Position=", "").replace("x:", "").replace("y:", "").replace("z:", "").split()
                        latest_state['x'] = float(parts[0])
                        latest_state['y'] = float(parts[1])
                        latest_state['z'] = float(parts[2])
                    elif line.startswith("Velocity="):
                        parts = line.replace("Velocity=", "").replace("x:", "").replace("y:", "").replace("z:", "").split()
                        latest_state['vx'] = float(parts[0])
                        latest_state['vy'] = float(parts[1])
                        latest_state['vz'] = float(parts[2])
                    elif line.startswith("Checkpoint="):
                        latest_state['cp'] = int(line.split('Checkpoint=')[1])
                        # writer.writerow([time.time(), latest_state['speed'], latest_state['x'], latest_state['y'], latest_state['z'], latest_state['vx'], latest_state['vy'], latest_state['vz']])
                    elif line.startswith("STOP"):
                        print("Received STOP command, shutting down...")
                        shutdown_event.set()
                        break
            except Exception as e:
                print("Error reading data:", e)
                break
        # Save the statesInEpoch to a file
        conn.close()
            #model = train(model)
def genModel():
    # inp_image = tf.keras.Input(shape=(224, 224, 3))
    # x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inp_image)
    normalizer = tf.keras.layers.Normalization(axis=-1)
    inp_telemetry = tf.keras.Input(shape=(8,))
    x = normalizer(inp_telemetry)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs=inp_telemetry, outputs=x)
    model.compile(optimizer='adam')
    return model, normalizer

def scale_state(state):
    # Scale the state values to a range suitable for the model
    state['speed'] = state['speed'] / 100.0  # Assuming speed is in km/h
    state['x'] = (state['x'] + 1000) / 2000.0  # Assuming x is in a range of -100 to 100
    state['y'] = (state['y']) / 100.0  # Assuming y is in a range of -100 to 100
    state['z'] = (state['z'] + 1000) / 2000.0   # Assuming z is in a range of -50 to 50
    state['vx'] /= 100.0                     # Normalize velocity components
    state['vy'] /= 100.0
    state['vz'] /= 100.0
    state['cp'] = int(state['cp'] / 20)
    return state

def inference(model):
    # This function will run in a separate thread to control the car based on the game state
    # It will read the latest state and use the model to decide actions
    global episode
    # global latest_state
    prev_state = latest_state.copy()
    for i in range(5):
        print(i)
        start_time = time.time()
        episode_data = []
        while (time.time() - start_time) < 10:
            time.sleep(3)
            #model.load_weights('model_weights.keras')
            state = scale_state(latest_state)
            state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['vx'], state['vy'], state['vz'], state['cp']]])
            #print(latest_state, state_vec)
            logits = model(state_vec)
            probs = tf.nn.sigmoid(logits)[0].numpy()
            action = (np.random.rand(4) < probs).astype(float)
            reward = compute_reward(prev_state, latest_state)
            prev_state = latest_state.copy()
            episode_data.append({'state':state_vec[0],'action': action,'reward': reward})
            #print(f"Action: {action}, Speed: {latest_state['speed']}, Position: ({latest_state['x']}, {latest_state['y']}, {latest_state['z']})")
            if action[0]: keyboard.press('w')  
            else: keyboard.release('w')
            if action[1]: keyboard.press('a')  
            else: keyboard.release('a')
            if action[2]: keyboard.press('s')  
            else: keyboard.release('s')
            if action[3]: keyboard.press('d')  
            else: keyboard.release('d')
            time.sleep(0.05)

        with open(f'{episode}statesInEpoch.pkl', 'wb') as f:
            pickle.dump(episode_data, f)
        episode += 1
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
    # shutdown_event.set()

if __name__ == "__main__":
    model, normalizer = genModel()
    read = Thread(target=read_game_state)
    read.start()
    connection_event.clear()
    connection_event.wait()
    dummy = np.zeros((1, 8), dtype=np.float32)
    model.predict(dummy)
    for _ in range(100):
        shutdown_event.clear()
        inf = Thread(target=inference, args=(model,))
        inf.start()
        # Wait for threads to finish before starting the next iteration
        inf.join()
        model = train(model, normalizer)
    read.join()