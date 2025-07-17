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
import dxcam, cv2
from collections import deque


k = 15
frame_buffer = deque(maxlen=k)

region = (640, 300, 1920, 1200)
camera = dxcam.create(output_idx=0, output_color="GRAY", region=region)

camera.start()
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



# telemetry data used in the model and for the rewards
# speed, position and current cp
latest_state = {'ts':0,'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'cp': 0.0}
episode = 0

# captures screen and as a grayscale and resizes it to be a smaller image to be used in a CNN
def get_screenshot():
    screenshot = camera.get_latest_frame()
    img = screenshot[:, :, :1]  
    img = cv2.resize(img, (200, 100))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Computes reward based on vehicle speed and checkpoint progress.
def compute_reward(prev_state, current_state, alpha=1.0, gamma=5.0):
    speed = current_state['speed'] * 100
    cp_diff = (current_state['cp'] - prev_state['cp']) * 5
    cp_reward = gamma * max(cp_diff, 0)

    return alpha * speed + cp_reward

# Computes the policy gradient loss from predicted logits and taken actions.
def policy_loss(logits, actions, rewards):
    logp = actions * tf.math.log(logits + 1e-10) + (1 - actions) * tf.math.log(1 - logits + 1e-10)
    return -tf.reduce_mean(tf.reduce_sum(logp, axis=1) * rewards)

# Calculates discounted returns to use as advantages for training.
def compute_advantage(rewards, gamma=0.99):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        advantages[t] = running_add
    return advantages

# Loads past episodes and trains the model in batches.
def train(model, init=False):
    BATCH_SIZE = 16
    def train_batch(images, states, move_act, turn_act, rewards):
        for i in range(0, len(images), BATCH_SIZE):
            image_batch = images[i:i+BATCH_SIZE]
            state_batch = states[i:i+BATCH_SIZE]
            move_batch = move_act[i:i+BATCH_SIZE]
            turn_batch = turn_act[i:i+BATCH_SIZE]
            adv_batch = rewards[i:i+BATCH_SIZE]

            with tf.GradientTape() as tape:
                move, turn = model([image_batch], training=True)
                move_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=move_batch, logits=move)

                turn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=turn_batch, logits=turn)
                weighted_move_loss = (move_loss) * tf.expand_dims(adv_batch, axis=1)
                weighted_turn_loss = (turn_loss) * tf.expand_dims(adv_batch, axis=1)
                pg_move_loss = tf.reduce_mean(weighted_move_loss + weighted_turn_loss)

                probs = tf.nn.sigmoid(move)
                ent_move = -tf.reduce_mean(probs * tf.math.log(probs + 1e-10)) + tf.reduce_mean((1 - probs) * tf.math.log(1 - probs + 1e-10))
                probs = tf.nn.sigmoid(turn)
                ent_turn = -tf.reduce_mean(probs * tf.math.log(probs + 1e-10)) + tf.reduce_mean((1 - probs) * tf.math.log(1 - probs + 1e-10))
                loss = pg_move_loss - 0.05 * (ent_move + ent_turn)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Episode {0}, Loss: {loss.numpy()}")

    # uses past recordings to boost the initial learning a bit though never did what I hoped it would
    # it has been removed in more recent times
    if init:
        for ep in range(1):
            for i in ['curve.pkl' , 'turn.pkl', 'long.pkl']: #,'straight.pkl'
                with open(f'{ep}{i}', 'rb') as f:
                    statesInEpoch = pickle.load(f)
                states = np.stack([state['state'] for state in statesInEpoch], axis=0)
                images = np.concatenate([state['image'] for state in statesInEpoch], axis=0)
                move_act = np.stack([state['move'] for state in statesInEpoch], axis=0).astype(np.int32)
                turn_act = np.stack([state['turn'] for state in statesInEpoch], axis=0).astype(np.int32)
                rewards = np.array([state['reward'] for state in statesInEpoch], dtype=np.float32)
                train_batch(images, states, move_act, turn_act, rewards)
    else:
        for ep in range(episode):
            print(ep)
            with open(f'{ep}statesInEpoch.pkl', 'rb') as f:
                statesInEpoch = pickle.load(f)
            states = np.stack([state['state'] for state in statesInEpoch], axis=0)
            images = np.concatenate([state['image'] for state in statesInEpoch], axis=0)
            move_act = np.stack([state['move'] for state in statesInEpoch], axis=0).astype(np.int32)
            turn_act = np.stack([state['turn'] for state in statesInEpoch], axis=0).astype(np.int32)
            rewards = np.array([state['reward'] for state in statesInEpoch], dtype=np.float32)
            train_batch(images, states, move_act, turn_act, rewards)
        
    model.save('model_weights.keras', save_format='keras')
    return model

# Listens to socket input and updates latest telemetry state with screenshots.
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
                        latest_state['pic'] = get_screenshot()
            except Exception as e:
                print("Error reading data:", e)
                break
        conn.close()

# Constructs the Keras CNN model using ConvLSTM layers.
# switched from tensorflow to pytorch after spending countless hours 
# trying to get tensorflow to load everything onto my GPU
def genModel():
    inp_image = tf.keras.Input(shape=(15, 100, 200, 1))
    x = tf.keras.layers.ConvLSTM2D(32, (5, 5), padding='same')(inp_image)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    img_feat = tf.keras.layers.GlobalAveragePooling2D()(x)

    #normalizer = tf.keras.layers.Normalization(axis=-1)
    # inp_telemetry = tf.keras.Input(shape=(5,))
    # telem_feat = tf.keras.layers.Dense(64, activation='relu')(inp_telemetry)
    # merge = tf.keras.layers.Concatenate()([img_feat, telem_feat])
    #x = normalizer(inp_telemetry)
    x = tf.keras.layers.Dense(512, activation='relu')(img_feat)
    # x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.Activation('relu')(x)
    move = tf.keras.layers.Dense(3, name='move')(x) # 0 = forward, 1 = backward, 2 = nothing
    lat = tf.keras.layers.Dense(3, name='turn')(x) # 0 = left, 1 = right, 2 = nothing
    model = tf.keras.Model(inputs=[inp_image], outputs=[move, lat])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

# Normalizes telemetry data for model input.
def scale_state(state):
    state['speed'] = (state['speed']) / 100.0
    state['x'] = (state['x']) / 1000.0
    state['y'] = (state['y']) / 100.0
    state['z'] = (state['z']) / 1000.0
    state['cp'] = int(state['cp'])/ 5.0
    return state

# Runs game simulation using the trained model and logs results.
def inference(model):
    global episode
    episode = 0
    for i in range(5):
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        print(i)

        start_time = time.time()
        episode_data = []

        with state_lock:
            prev_state = scale_state(latest_state)
            frame_buffer.extend([latest_state['pic']] * k)
        
        cp_time = time.time()
        cp_num = 0

        while (time.time() - start_time) < 35 and (time.time() - cp_time) < 5:
            if time.time() - start_time < 2:
                cp_time = time.time() + 2
                time.sleep(2)
            
            if latest_state['ts'] < prev_state['ts']:
                continue

            with state_lock:
                state = scale_state(latest_state).copy()

            if state['cp'] > cp_num:
                cp_num = state['cp']
                cp_time = time.time()

            state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['cp']]])
            image = state['pic']
            frame_buffer.append(image)
            stacked = np.stack([f[0] for f in frame_buffer], axis=0)
            stacked = np.expand_dims(stacked, 0)
            move, turn = model([stacked])
            probs_move = tf.nn.softmax(move)[0].numpy()
            probs_turn = tf.nn.softmax(turn)[0].numpy()
            epsilon = 0.05
            rand = np.random.choice([0,1], p=[epsilon, 1-epsilon])

            if rand == 1:
                move_action = np.argmax(probs_move)
                turn_action = np.argmax(probs_turn)
            else:
                move_action = np.random.choice([0, 1, 2], p=probs_move)
                turn_action = np.random.choice([0, 1, 2], p=probs_turn)

            reward = compute_reward(prev_state, state)
            if move_action != 2:
                reward += 0.02

            prev_state = state.copy()
            episode_data.append({'state':state_vec[0], 'image': stacked,'move': move_action, 'turn': turn_action,'reward': reward})
            
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

        with open(f'{episode}statesInEpoch.pkl', 'wb') as f:
            pickle.dump(episode_data, f)

        episode += 1
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
    

if __name__ == "__main__":
    model = genModel()
    read = Thread(target=read_game_state)
    read.start()
    connection_event.clear()
    connection_event.wait()
    image = get_screenshot()
    frame_buffer.extend([image] * k)
    stacked = np.stack([f[0] for f in frame_buffer], axis=0)
    stacked = np.expand_dims(stacked, 0)
    model.predict(stacked) # initializes weights
    model = train(model, True)

    for _ in range(1000):
        shutdown_event.clear()
        inf = Thread(target=inference, args=(model,))
        inf.start()
        inf.join()
        model = train(model)

    end_event.set()
    camera.stop()
    read.join()