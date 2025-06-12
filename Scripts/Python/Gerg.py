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
end_event = threading.Event()
connection_event = threading.Event()

latest_state = {'ts':0,'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'cp': 0}
episode = 0

def compute_reward(prev_state, current_state, alpha=3.0, beta=0.0, gamma=0.0, forw=1.0, sid=1.0):
    speed = current_state['speed'] * 100
    #acceleration = (current_state['speed'] - prev_state['speed']) * 100
    delta_cp = (current_state['cp']) * 20
    #forward = (current_state['x'] - prev_state['x']) * 100
    #side = (current_state['z'] - prev_state['z']) * 1000
    #print("speed:", speed, "acceleration:", acceleration, "delta_cp:", delta_cp, "forward:", forward, "side:", side)
    #if acceleration < 0 and speed > 0:
        #beta = 5.0
    # print("prev_state:", prev_state)
    # print("current_state:", current_state)
    # print("speed:", speed, "acceleration:", acceleration, "delta_cp:", delta_cp)
    #print("reward:", alpha * speed + beta * acceleration + gamma * delta_cp + forw * forward) #, "speed:", speed, "delta_cp:", delta_cp
    return alpha * speed + gamma * delta_cp # + beta * acceleration  + forw * forward

def policy_loss(logits, actions, rewards):
    logp = actions * tf.math.log(logits + 1e-10) + (1 - actions) * tf.math.log(1 - logits + 1e-10)
    return -tf.reduce_mean(tf.reduce_sum(logp, axis=1) * rewards)

def compute_advantage(rewards, gamma=0.99):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        advantages[t] = running_add
    b = np.mean(advantages)
    # advantages -= b
    #advantages /= (np.std(advantages) + 1e-8)
    return advantages

def train(model, episodes=50):
    for ep in range(episode):
        print(ep)
        with open(f'{ep}statesInEpoch.pkl', 'rb') as f:
            statesInEpoch = pickle.load(f)
        states = np.stack([state['state'] for state in statesInEpoch], axis=0).astype(np.float32)
        #actions = np.stack([state['action'] for state in statesInEpoch], axis=0).astype(np.float32)
        move_act = np.stack([state['move'] for state in statesInEpoch], axis=0).astype(np.int32)
        turn_act = np.stack([state['turn'] for state in statesInEpoch], axis=0).astype(np.int32)
        rewards = np.array([state['reward'] for state in statesInEpoch], dtype=np.float32)
        initial_state = states[1]
        final_state = states[-1]
        x_total = final_state[0] - initial_state[0]
        cp_gain = final_state[7] - initial_state[7]
        print(f"Episode {ep}, X Total: {1000*x_total}, CP Gain: {20*cp_gain}")
        final_bonus = 1000 * x_total + 20 * cp_gain
        rewards = [reward + final_bonus for reward in rewards]
        
        #print(states[10:20])
        #print(actions[10:20])
        #normalizer.adapt(states)
        advantages = rewards.copy()
        #advantages = compute_advantage(rewards)
        with tf.GradientTape() as tape:
            move, turn = model(states, training=True)
            move_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=move_act, logits=move)

            turn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=turn_act, logits=turn)
            weighted_loss = (turn_loss + move_loss) * tf.expand_dims(advantages, axis=1)
            pg_move_loss = tf.reduce_mean(weighted_loss)

            probs = tf.nn.sigmoid(move)
            ent_move = -tf.reduce_mean(probs * tf.math.log(probs + 1e-10)) + tf.reduce_mean((1 - probs) * tf.math.log(1 - probs + 1e-10))
            probs = tf.nn.sigmoid(turn)
            ent_turn = -tf.reduce_mean(probs * tf.math.log(probs + 1e-10)) + tf.reduce_mean((1 - probs) * tf.math.log(1 - probs + 1e-10))
            loss = pg_move_loss - 0.02 * (ent_move + ent_turn)
            #logits = tf.nn.sigmoid(logits)
            
            #loss = policy_loss(logits, actions, rewards)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Episode {ep}, Loss: {loss.numpy()}")
    
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
        prev_time = 0
        while not end_event.is_set():
            try:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break
                if data.strip().splitlines()[0].startswith("TS="):
                    print(data.strip().splitlines()[0].replace('TS=', '').split()[0])
                    cur_time = float(data.strip().splitlines()[0].replace('TS=', '').split()[0])
                    if cur_time < prev_time:
                        print("Received old data, skipping...")
                        continue
                    prev_time = cur_time
                #print("received data:", data)
                #begin = time.time()
                #print("begin: ",time.time())
                for line in data.strip().splitlines():
                    # if line.startswith("Speed="):
                    #     latest_state['speed'] = float(line.split('=')[1])
                    # elif line.startswith("Position="):
                    #     parts = line.replace("Position=", "").replace("x:", "").replace("y:", "").replace("z:", "").split()
                    #     latest_state['x'] = float(parts[0])
                    #     latest_state['y'] = float(parts[1])
                    #     latest_state['z'] = float(parts[2])
                    # elif line.startswith("Velocity="):
                    #     parts = line.replace("Velocity=", "").replace("x:", "").replace("y:", "").replace("z:", "").split()
                    #     latest_state['vx'] = float(parts[0])
                    #     latest_state['vy'] = float(parts[1])
                    #     latest_state['vz'] = float(parts[2])
                    # elif line.startswith("Checkpoint="):
                    #     latest_state['cp'] = int(line.split('Checkpoint=')[1])
                        # writer.writerow([time.time(), latest_state['speed'], latest_state['x'], latest_state['y'], latest_state['z'], latest_state['vx'], latest_state['vy'], latest_state['vz']])
                    if line.startswith("STOP"):
                        print("Received STOP command, shutting down...")
                        shutdown_event.set()
                        break
                    else:
                        line = line.split(',')
                        #print(len(line), line)
                        if len(line) != 9:
                            continue
                        #print(line)
                        latest_state['ts'] = int(line[0])
                        latest_state['speed'] = float(line[1])
                        latest_state['x'] = float(line[2])
                        latest_state['y'] = float(line[3])
                        latest_state['z'] = float(line[4])
                        latest_state['vx'] = float(line[5])
                        latest_state['vy'] = float(line[6])
                        latest_state['vz'] = float(line[7])
                        latest_state['cp'] = int(line[8])
                #print("end: ",time.time() - begin)
            except Exception as e:
                print("Error reading data:", e)
                break
            # Save the statesInEpoch to a file
        conn.close()
                #model = train(model)
def genModel():
    # inp_image = tf.keras.Input(shape=(224, 224, 3))
    # x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inp_image)
    #normalizer = tf.keras.layers.Normalization(axis=-1)
    inp_telemetry = tf.keras.Input(shape=(8,))
    #x = normalizer(inp_telemetry)
    x = tf.keras.layers.Dense(1024, activation='relu')(inp_telemetry)
    #x = tf.keras.layers.Dropout(0.5)(x)
    #x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    #x = tf.keras.layers.ReLU()(x)
    move = tf.keras.layers.Dense(3, name='move')(x) # 0 = forward, 1 = backward, 2 = nothing
    lat = tf.keras.layers.Dense(3, name='turn')(x) # 0 = left, 1 = right, 2 = nothing
    model = tf.keras.Model(inputs=inp_telemetry, outputs=[move, lat])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    return model

def scale_state(state):
    # Scale the state values to a range suitable for the model
    state['speed'] = (state['speed']) / 100.0  # Assuming speed is in a range of 0 to 100
    state['x'] = (state['x']) / 1000.0  # Assuming x is in a range of -100 to 100
    state['y'] = (state['y']) / 100.0  # Assuming y is in a range of -100 to 100
    state['z'] = (state['z']) / 1000.0   # Assuming z is in a range of -50 to 50
    state['vx'] = (state['vx']) / 100.0  # Assuming vx is in a range of -100 to 100
    state['vy'] = (state['vy']) / 100.0  # Assuming vy is in a range of -100 to 100
    state['vz'] = (state['vz']) / 100.0  # Assuming vz is in a range of -100 to 100
    state['cp'] = int(state['cp'])/ 20.0
    return state

def inference(model):
    # This function will run in a separate thread to control the car based on the game state
    # It will read the latest state and use the model to decide actions
    global episode
    episode = 0
    # global latest_state
    for i in range(5):
        print(i)
        start_time = time.time()
        episode_data = []
        prev_state = scale_state(latest_state)
        print(prev_state)
        cp_time = time.time()
        cp_num = 0
        while (time.time() - start_time) < 50 and (time.time() - cp_time) < 10: #add thing that after 8 seconds of no new cp end eapisode
            if time.time() - start_time < 2:
                cp_time = time.time() + 2
                time.sleep(2)
            if latest_state['ts'] < prev_state['ts']:

                #time.sleep(1.0/60.0)
                continue
            #model.load_weights('model_weights.keras')
            state = scale_state(latest_state).copy()
            if state['cp'] > cp_num:
                cp_num = state['cp']
                cp_time = time.time()
            state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['vx'], state['vy'], state['vz'], state['cp']]])
            #print(prev_state, latest_state)
            move, turn = model(state_vec)
            probs_move = tf.nn.softmax(move)[0].numpy()
            probs_turn = tf.nn.softmax(turn)[0].numpy()
            if i == 1:
                move_action = np.argmax(probs_move)
                turn_action = np.argmax(probs_turn)
            else:
                move_action = np.random.choice([0, 1, 2], p=probs_move)
                turn_action = np.random.choice([0, 1, 2], p=probs_turn)
            # logits = model(state_vec)
            # probs = tf.nn.sigmoid(logits)[0].numpy()
            # action_idx = 
            # action = (probs > 0.5).astype(float)
            #action = np.random.choice([0, 1], size=3, p=probs)
            #action = (np.random.rand(3) < probs).astype(float)
            reward = compute_reward(prev_state, state)
            #print(move_action, turn_action)
            prev_state = state.copy()
            episode_data.append({'state':state_vec[0],'move': move_action, 'turn': turn_action,'reward': reward})
            #print(f"Action: {action}, Speed: {latest_state['speed']}, Position: ({latest_state['x']}, {latest_state['y']}, {latest_state['z']})")
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
            # if action[0]: keyboard.press('w')
            # else: keyboard.release('w')
            # if action[1]: keyboard.press('s') 
            # else: keyboard.release('s')
            # if action[2]: keyboard.press('a')  
            # else: keyboard.release('a')
            # if action[3]: keyboard.press('d')  
            # else: keyboard.release('d')
            time.sleep(1.0/60.0)

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
    
    # shutdown_event.set()

if __name__ == "__main__":
    model = genModel()
    read = Thread(target=read_game_state)
    read.start()
    connection_event.clear()
    connection_event.wait()
    dummy = np.zeros((1, 8), dtype=np.float32)
    model.predict(dummy)
    for _ in range(1000):
        shutdown_event.clear()
        inf = Thread(target=inference, args=(model,))
        inf.start()
        # Wait for threads to finish before starting the next iteration
        inf.join()
        model = train(model)
    end_event.set()
    read.join()