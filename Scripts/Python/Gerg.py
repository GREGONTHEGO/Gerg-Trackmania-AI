from pynput.keyboard import Key, Controller
import time
import socket
from threading import Thread
import threading
import tensorflow as tf
import numpy as np
import pickle

HOST = '127.0.0.1'
PORT = 5055
keyboard = Controller()
shutdown_event = threading.Event()
end_event = threading.Event()
connection_event = threading.Event()

latest_state = {'ts':0,'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'cp': 0}
episode = 0

# Computes the reward for the current state just based on speed
def compute_reward(current_state, alpha=3.0):
    speed = current_state['speed'] * 100

    return alpha * speed

def policy_loss(logits, actions, rewards):
    logp = actions * tf.math.log(logits + 1e-10) + (1 - actions) * tf.math.log(1 - logits + 1e-10)
    return -tf.reduce_mean(tf.reduce_sum(logp, axis=1) * rewards)

def compute_advantage(rewards, gamma=0.99):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        advantages[t] = running_add

    return advantages


def train(model):
    for ep in range(episode):
        print(ep)
        with open(f'{ep}statesInEpoch.pkl', 'rb') as f:
            statesInEpoch = pickle.load(f)
        states = np.stack([state['state'] for state in statesInEpoch], axis=0).astype(np.float32)
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

        advantages = rewards.copy()
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

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Episode {ep}, Loss: {loss.numpy()}")
        
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
                for line in data.strip().splitlines():
                    if line.startswith("STOP"):
                        print("Received STOP command, shutting down...")
                        shutdown_event.set()
                        break
                    else:
                        line = line.split(',')
                        if len(line) != 9:
                            continue
                        latest_state['ts'] = int(line[0])
                        latest_state['speed'] = float(line[1])
                        latest_state['x'] = float(line[2])
                        latest_state['y'] = float(line[3])
                        latest_state['z'] = float(line[4])
                        latest_state['vx'] = float(line[5])
                        latest_state['vy'] = float(line[6])
                        latest_state['vz'] = float(line[7])
                        latest_state['cp'] = int(line[8])
            except Exception as e:
                print("Error reading data:", e)
                break
        conn.close()
def genModel():
    inp_telemetry = tf.keras.Input(shape=(8,))
    x = tf.keras.layers.Dense(1024, activation='relu')(inp_telemetry)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    move = tf.keras.layers.Dense(3, name='move')(x) # 0 = forward, 1 = backward, 2 = nothing
    lat = tf.keras.layers.Dense(3, name='turn')(x) # 0 = left, 1 = right, 2 = nothing
    model = tf.keras.Model(inputs=inp_telemetry, outputs=[move, lat])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    return model

def scale_state(state):
    state['speed'] = (state['speed']) / 100.0
    state['x'] = (state['x']) / 1000.0
    state['y'] = (state['y']) / 100.0
    state['z'] = (state['z']) / 1000.0
    state['vx'] = (state['vx']) / 100.0
    state['vy'] = (state['vy']) / 100.0
    state['vz'] = (state['vz']) / 100.0
    state['cp'] = int(state['cp'])/ 20.0
    return state

def inference(model):
    global episode
    episode = 0
    for i in range(5):
        print(i)
        start_time = time.time()
        episode_data = []
        prev_state = scale_state(latest_state)
        print(prev_state)
        cp_time = time.time()
        cp_num = 0
        while (time.time() - start_time) < 50 and (time.time() - cp_time) < 10:
            if time.time() - start_time < 2:
                cp_time = time.time() + 2
                time.sleep(2)
            if latest_state['ts'] < prev_state['ts']:
                continue
            state = scale_state(latest_state).copy()
            if state['cp'] > cp_num:
                cp_num = state['cp']
                cp_time = time.time()
            state_vec = np.array([[state['speed'], state['x'], state['y'], state['z'], state['vx'], state['vy'], state['vz'], state['cp']]])
            move, turn = model(state_vec)
            probs_move = tf.nn.softmax(move)[0].numpy()
            probs_turn = tf.nn.softmax(turn)[0].numpy()
            if i == 1:
                move_action = np.argmax(probs_move)
                turn_action = np.argmax(probs_turn)
            else:
                move_action = np.random.choice([0, 1, 2], p=probs_move)
                turn_action = np.random.choice([0, 1, 2], p=probs_turn)
            reward = compute_reward(prev_state, state)
            prev_state = state.copy()
            episode_data.append({'state':state_vec[0],'move': move_action, 'turn': turn_action,'reward': reward})

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
        inf.join()
        model = train(model)
    end_event.set()
    read.join()