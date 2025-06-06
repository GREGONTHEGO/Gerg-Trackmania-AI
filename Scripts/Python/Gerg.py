from pynput.keyboard import Key, Controller
import time
import os
import socket
from threading import Thread
import csv
import threading
import tensoflow as tf

HOST = '127.0.0.1'
PORT = 5055
keyboard = Controller()
shutdown_event = threading.Event()

latest_state = {'speed': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
def read_game_state():
    global latest_state
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print("listening on", HOST, PORT)
        conn, addr = s.accept()
        print(f"connection from {addr}")

        with open('car_state.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp','speed', 'x', 'y', 'z'])

            while not shutdown_event.is_set():
                try:
                    data = conn.recv(1024).decode('utf-8')
                    if not data:
                        break
                    print("received data:", data)
                    #state = data.strip().split(',')
                    for line in data.strip().splitlines():
                        if line.startswith("Speed="):
                            latest_state['speed'] = float(line.split('=')[1])
                        elif line.startswith("Position="):
                            parts = line.replace("Position=", "").replace("x:", "").replace("y:", "").replace("z:", "").split()
                            latest_state['x'] = float(parts[0])
                            latest_state['y'] = float(parts[1])
                            latest_state['z'] = float(parts[2])

                            writer.writerow([time.time(), latest_state['speed'], latest_state['x'], latest_state['y'], latest_state['z']])
                        elif line.startswith("STOP"):
                            print("Received STOP command, shutting down...")
                            shutdown_event.set()
                            break
                except Exception as e:
                    print("Error reading data:", e)
                    break
def control_loop():
    global latest_state
    while not shutdown_event.is_set():
        speed = latest_state['speed']
        #print(speed)
        if speed < 50:
            keyboard.press('w')
        else:
            keyboard.release('w')
        
        time.sleep(0.05)

if __name__ == "__main__":
    Thread(target=read_game_state).start()
    Thread(target=control_loop).start()