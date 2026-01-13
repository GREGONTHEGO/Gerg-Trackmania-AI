# Project Log: Reinforcement Learning in Trackmania

This document outlines the major stages of developing my reinforcement learning system which is designed to control a vehicle in Trackmania (2020). Each subsection reflects iterative improvements, experiments, and lessons learned across areas including telemetry extraction, model design, reward engineering, and architectural changes.

## Table of Contents
- [Basic DNN](/EXPERIMENTS.md#basic-dnn--gamestatesonlypy)
- [CNN with TensorFlow](/EXPERIMENTS.md#cnn-with-tensorflow--cnnpy)
- [CNN with PyTorch](/EXPERIMENTS.md#cnn-with-pytorch--cnntorchpy)
- [LIDAR with Softmax Policy](/EXPERIMENTS.md#lidar-with-softmax-policy--lidarpy)
- [LIDAR with Gaussian Policy](/EXPERIMENTS.md#lidar-with-gaussian-policy--lidargausspy)
- [Ghost Trajectory Agent](/EXPERIMENTS.md#ghost-trajectory-agent)


## [Basic DNN](/README.md):  [(gameStateOnly.py)](/Scripts/Python/gameStateOnly.py)

A simple deep neural network built with TensorFlow that relies solely on telemetry inputs, without any visual information. It uses values such as speed, position, velocity, and checkpoint count to determine the next action. While limited in perception compared to image-based models, it serves as a useful baseline for understanding the basics of a reinforcement learning model in a game environment.

**Experiment 1: Retrieving Game Data and Controlling the Vehicle**

The initial goal was to extract real-time game data from Trackmania (2020) and use that data to control the vehicle using Python. This was accomplished through OpenPlanet, a scripting platform for Trackmania. By writing a plugin in AngelScript, I was able to access internal game state variables (speed, position, RPM, gear, etc.) and stream them via a TCP socket to a Python client.

Simultaneously, the Python client was set up to send control commands (throttle, brake, steering) back to the game. I used the pynput library to simulate keyboard presses, mapping the model’s decisions to 'W', 'A', 'S', 'D' keys. A critical challenge here was latency; ensuring that the data extraction, model inference, and key-press execution happened fast enough to drive the car effectively at high speeds.

**Experiment 2: Simple Reward**

https://github.com/user-attachments/assets/5507724a-d6ce-42fa-a510-b82c61f4d2be

With the communication pipeline established, I implemented a basic Deep Neural Network (DNN) using TensorFlow. The input layer took the telemetry data, and the output layer provided action probabilities.

The first reward function was straightforward:
- **Reward = Speed × 1.0**

The hypothesis was that rewarding speed would naturally encourage the car to move forward and avoid crashing (which reduces speed to zero). While the car did learn to hold the accelerator, it struggled with corners, often crashing into walls because it didn't understand the concept of steering to maintain speed over time. It simply maximized instantaneous speed.

**Experiment 3: Complex Reward**

To address the shortcomings of the simple reward, I designed a more sophisticated reward function. This version incorporated:
- **Checkpoint Reward:** A large bonus (+100) for passing a checkpoint.
- **Velocity Vector Analysis:** Rewarding velocity specifically in the direction of the track (using position deltas) rather than just raw scalar speed.
- **Penalties:** Negative rewards for wall collisions or dropping below a certain speed threshold.

This iteration improved performance significantly. The agent began to associate "progress" (checkpoints) with high rewards. However, tuning the balance between the speed reward and the checkpoint bonus proved difficult. If the checkpoint bonus was too high, the agent would sometimes drive erratically just to hit the checkpoint, ignoring the optimal racing line.


## [CNN with TensorFlow](/README.md):  [(cnn.py)](/Scripts/Python/cnn.py)

A TensorFlow model that uses 2D Convolutional Long Short Term Memory (ConvLSTM2D) for the CNN layers.

**Experiment 4: Vision**

[![YouTube Video](https://img.youtube.com/vi/-kLVGGpw-KU/0.jpg)](https://youtube.com/watch?v=-kLVGGpw-KU)

Transitioning from telemetry-only to visual input, I integrated a Convolutional Neural Network (CNN). Using DXCam, I captured the game window in real-time, converted the frames to grayscale, and resized them to reduce computational load.

The architecture used `ConvLSTM2D` layers to process a stack of frames. The idea was that a single frame gives position, but a stack of frames gives the model a sense of motion and acceleration (temporal dynamics).

*Challenge:* The primary bottleneck became the training loop speed. Capturing images, processing them with TensorFlow on the CPU (or inefficient GPU transfer), and running backpropagation resulted in a low frame rate for the agent. This latency made it impossible for the agent to react to sudden turns or obstacles in real-time.


## [CNN with PyTorch](/README.md):  [(cnnTorch.py)](/Scripts/Python/cnnTorch.py)

A PyTorch convolutional neural network that combines grayscale screenshots and telemetry data to predict actions.

**Experiment 5: PyTorch**

Due to the performance issues with TensorFlow in this specific real-time setup, I migrated the project to PyTorch. PyTorch’s dynamic computation graph and efficient tensor operations allowed for faster inference and training loops.

I also optimized the screen capture pipeline, ensuring that frame buffering and tensor conversion happened asynchronously where possible. This reduction in latency allowed the CNN agent to actually "see" a corner approaching and react in time. The model architecture remained a CNN processing stacked frames, but the improved throughput allowed for more complex experimentation with hyperparameters.


## [LIDAR with Softmax Policy](/README.md):  [(lidar.py)](/Scripts/Python/lidar.py)

An experiment using “LIDAR” with a PyTorch model that uses discrete softmax outputs for the two sets of three values represented by movement and turning.

**Experiment 6: LIDAR**

[![YouTube Video](https://img.youtube.com/vi/FMvDgTzFy70/0.jpg)](https://youtube.com/watch?v=FMvDgTzFy70)

Processing full images is computationally expensive and can include noise (like track decorations or lighting changes) that is irrelevant to driving physics. To simplify the input while retaining spatial awareness, I implemented a simulated LIDAR system.

*Implementation:*
- I wrote a script to cast rays from the car’s position in the captured image.
- By detecting the contrast between the track (usually grey/black) and the walls/grass, the system calculated the distance to the nearest obstacle along several angles.

This "LIDAR" data [d1, d2, d3...] was fed into the network instead of the raw pixel data. This drastically reduced the input dimension, leading to faster training. The model used a Softmax policy to classify actions into discrete categories (e.g., Turn Left, Straight, Turn Right). The agent became much better at wall avoidance, as the "distance to wall" was now a direct numerical input feature rather than a feature that had to be learned from pixels.


## [LIDAR with Gaussian Policy](/README.md):  [(lidarGauss.py)](/Scripts/Python/lidarGauss.py)

The most recent experiment in the series uses “LIDAR” with Gaussian policy and is built using PyTorch and a Gaussian policy. The output of this model is three values that are sent through the tanh function to return a real number on the scale of [-1,1]. The model produces three values representing movement decisions. These outputs are mapped to game controls: forward, backward, and turning. Compared to its predecessor (lidar.py), aside from changing the policy, this model features a larger architecture and updated reward functions.

**Experiment 7: Emulate Others**

Despite this, the model often prioritized speed-based rewards over collision penalties, leading to reckless behavior such as deliberate crashes for short-term gain.
To try something new, the model architecture was redesigned to incorporate:
- A Gaussian policy head for continuous action sampling, rather than discrete classification.
- A unified output structure with three action values, interpreted via tanh activation function (essentially sets the scale to [-1,1]). This replaced the earlier two-head system, which independently selected from {forward, backward, none} and {left, right, none}.

These changes were inspired by architectural strategies found in the [TMRL project](https://github.com/trackmania-rl/tmrl/tree/master), though my implementation uses my own approach to reward formulation. While the new design enables more expressive behavior, reward shaping remains a critical issue and continues to be a source of instability in training outcomes.

Some approaches rely on rewarding increased distance traveled along a predefined map path. However, this project intentionally avoids that method in order to prevent the model from overfitting to the predefined map path.


## [Ghost Trajectory Agent](/README.md): [(RL_Ghost_PPO_Agent.py)](/Scripts/Python/RL_Ghost_PPO_Agent.py)

**(Formerly `cnnTorch.py`)**

This experiment represents a shift from purely abstract reward shaping (speed/distance) to a demonstration-guided approach, reverting to a CNN architecture (rather than LIDAR) to capture the visual cues of the ghost car.

**Experiment 8: Follow the Ghost**

Previous models struggled to find the optimal racing line solely through exploration; they would often learn to drive safely but slowly, or fast but erratically. To solve this, I implemented a "Ghost" system, similar to the ghost cars found in racing games.

* **Mechanism:** The agent records the trajectory (positions and timestamps) of its best successful run. This becomes the "Ghost Path".
* **Reward Function:** Instead of just rewarding speed, the reward now includes a **Cross Track Error** penalty. We calculate the distance between the car's current position and the nearest point on the Ghost Path.
    * **High Reward:** High speed AND low distance to the Ghost Path.
    * **Penalty:** Deviating too far from the optimal line.

This allows the agent to iteratively improve. It starts by following a crude successful run (forced forward bias). Once it accidentally finds a faster line or hits a further checkpoint, that new run becomes the new Ghost, and the agent is then trained to replicate that superior behavior.

**Run Comparison:**

<img width="450" height="450" alt="best_path_plot" src="https://github.com/user-attachments/assets/3e230d46-90cc-46fe-aaff-31e89dd5715f" />

*Run _0: The agent survives by essentially dragging itself along the wall. This is indicated by the very smooth, unnatural line. While this minimizes crash penalties, it shows the model is optimizing for survival mechanics rather than proper driving physics.*

<img width="450" height="450" alt="best_path_plot_1" src="https://github.com/user-attachments/assets/c34c42b5-6240-4744-8084-242e21ccd9c6" />

*Run _1: The best overall run. This represents a significant improvement over the previous run; rather than simply sliding against the wall, the agent actively bounces off the walls and steers to maintain momentum. The jagged line indicates the agent is making active steering decisions to correct its path, demonstrating a better understanding of vehicle control.*
