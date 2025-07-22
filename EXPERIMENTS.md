# Project Log: Reinforcement Learning in Trackmania

This document outlines the five major stages of developing my reinforcement learning system which is designed to control a vehicle in Trackmania (2020). Each subsection reflects iterative improvements, experiments, and lessons learned across areas including telemetry extraction, model design, reward engineering, and architectural changes.

## Table of Contents
- [Basic DNN](/EXPERIMENTS.md#basic-dnn--gamestatesonlypy)
- [CNN with TensorFlow](/EXPERIMENTS.md#cnn-with-tensorflow--cnnpy)
- [CNN with PyTorch](/EXPERIMENTS.md#cnn-with-pytorch--cnntorchpy)
- [LIDAR with Softmax Policy](/EXPERIMENTS.md#lidar-with-softmax-policy--lidarpy)
- [LIDAR with Gaussian Policy](/EXPERIMENTS.md#lidar-with-gaussian-policy--lidargausspy)


## [Basic DNN](/README.md):  [(gameStateOnly.py)](/Scripts/Python/gameStateOnly.py)

A simple deep neural network built with TensorFlow that relies solely on telemetry inputs, without any visual information. It uses values such as speed, position, velocity, and checkpoint count to determine the next action. While limited in perception compared to image-based models, it serves as a useful baseline for understanding the basics of a reinforcement learning model in a game environment.

**Experiment 1: Retrieving Game Data and Controlling the Vehicle**

The initial goal was to extract real-time game data from Trackmania (2020) and use that data to control the vehicle using Python. This was accomplished through Openplanet, a modding framework that enables custom plugin development for Trackmania using AngelScript.
To implement the plugin, I wrote an 'info.toml' and a script located in 'plugins/GergBot/'. This plugin adds a menu item in Openplanet that launches a local socket server. The socket transmits the car's current speed, position, velocity, and checkpoint count on each update. However, there were two issues identified early:
1. No frame-rate control: Data was sent as fast as possible, leading to inconsistencies in synchronization and strange reward behaviors.

*Resolved by having the listener process only the most recent state per message batch.*

2. TOML structuring confusion: The 'meta' and 'script' sections were initially misused, which are crucial for proper plugin registration.

*Resolved by putting the dependencies in the scripts section and everything else in the meta section*

On the Python side, I created a function that runs two threads:
1. One that listens to the socket and updates a global 'latest_state' variable
2. The other makes movement decisions, such as if speed is below 50 m/s, it presses the gas; otherwise, it lets off.

During this phase, it was also discovered that 'pynput' holds keys down continuously until they are explicitly released, which causes unwanted keypress spam when switching windows.

**Experiment 2: Building the first Neural Network**

https://github.com/user-attachments/assets/5507724a-d6ce-42fa-a510-b82c61f4d2be

This stage focused on building the machine learning infrastructure. I restructured the Python codebase by adding:

- genModel(): A function to generate a neural network with 8 inputs, two hidden layers of 1024 neurons, and 4 outputs (forward, backward, left, and right)
- compute_reward(): speed-based reward
- policy_loss(): basic TensorFlow policy
- train(): train a given model on data from inference
- inference(): updated to call the function in real time and store the information

The inference function would gather state-action pairs, then train after a batch of runs. The first version succeeded in moving the car; however, due to a lack of synchronization between the plugin and Python threads, state sequences were mismatched. This mismatch has been fixed in more recent versions. (The fix was to take the most recent state each time the connection received 1024 bytes) The model, at one point, mistakenly learned that launching off a ramp (to achieve high speed briefly) was the optimal behavior.

**Experiment 3: Teaching the Model to Drive Straight**

The objective here was to get the model to learn basic straight-line driving. This turned out to be more difficult than expected due to reward misinterpretation:

- Penalizing deceleration caused the model to reverse and get stuck.
- Raw inputs with large position values resulted in spinning. (did not initialize the weights systematically)
- The model occasionally turned sharply and became stuck due to a lack of penalty in the lateral movement.

The reward function was refined to:

- Provide positive reinforcement for speed.
- Add a large bonus if a full episode performed better than prior ones.

Eventually, the model began to drive further and more consistently. A video demonstration is linked above. The speed value directly from Trackmania was positive when going forward or backward. However, when the velocity vector (in m/s) is projected onto the aim direction using a dot product, positive values represent forward movement, while negative values indicate reverse movement. Although the model successfully learned to apply forward motion consistently, it often failed to learn turning behaviors. The model converged toward always accelerating as the most reliable way to earn positive rewards, neglecting the importance of steering for faster map completion.

## [CNN with TensorFlow](/README.md):  [(cnn.py)](/Scripts/Python/cnn.py)

This experiment used a TensorFlow model that uses a ConvLSTM2D (ConvLSTM2D not in PyTorch). Although the model functions as intended, it was constrained to run on the CPU due to an incompatibility between the installed CUDA version and TensorFlow. Multiple attempts were made to enable GPU acceleration, but working configurations could not connect reliably to the local socket server.

**Experiment 4: Integrating Convolutional Neural Networks (CNNs)**

[![YouTube Video](https://img.youtube.com/vi/-kLVGGpw-KU/0.jpg)](https://youtube.com/watch?v=-kLVGGpw-KU)

This next stage introduced image-based perception and transitioned the project from TensorFlow to PyTorch for improved performance and GPU compatibility.
For visual input, the DXCam library was used to capture a grayscale subsection of the game screen in real time. Images were resized to 200x100 and stacked as sequences of recent frames, providing temporal context. Initially, 10-frame stacks were used, but this was later reduced to 5  frames to optimize relevance and memory usage.

## [CNN with PyTorch](/README.md):  [(cnnTorch.py)](/Scripts/Python/cnnTorch.py)

This experiment used a PyTorch convolutional neural network that combined grayscale screenshots and telemetry data to predict actions. Unlike the TensorFlow model, this version does not include Convolutional LSTM or 3D convolutions to process temporal information. Instead, image sequences are treated as additional channels in a single stacked input.

**Experiment 5: Switching libraries**

Due to limitations in TensorFlow's GPU integration and the lack of a native ConvLSTM2D in PyTorch, the project explored alternative solutions such as 3D convolutions and LSTMs followed by 2D convolution layers. These approaches were intended to enable the model to learn spatiotemporal features from visual input. However, 3D convolutions proved too computationally expensive, and the LSTM-based models failed to train effectively. As a result, the model was simplified to use only 2D convolutions.

After properly installing PyTorch with CUDA support, training performance increased significantly, reducing training times from around 5 minutes per episode to 5 seconds. This allowed for faster experimentation and tuning as well as faster inference.

To improve learning efficiency:
- The code base was modified to store multiple runs per training epoch, rather than training on the single best run.
- A best-run buffer was introduced, saving the highest-performing run observed so far. This "elite" run was combined with newly collected data from each epoch to reinforce previously successful strategies.

Despite these upgrades, limitations in training data diversity and screen variations posed challenges for the CNN to generalize across runs. The training images were captured from similar camera positions and mostly from early map segments, resulting in highly repetitive visual inputs. Without enough variation in training the CNN would not be very helpful in determining driving behavior.

## [LIDAR with Softmax Policy](/README.md):  [(lidar.py)](/Scripts/Python/lidar.py)

A PyTorch experiment that uses discrete softmax outputs for the two sets of three values represented by movement and turning. This approach emerged from the need for a more structured input format that could provide consistent, meaningful data compared to image-based models.

**Experiment 6: Transition to LIDAR-Based Perception and Reward Engineering**

[![YouTube Video](https://img.youtube.com/vi/FMvDgTzFy70/0.jpg)](https://youtube.com/watch?v=FMvDgTzFy70)

Given the resource constraints for training convolutional networks effectively, this phase explored an alternative input representation using simulated LIDAR-style data. This approach draws inspiration from Trackmania reinforcement learning projects, where rays are cast from the vehicle to estimate distances to nearby walls.
Initial attempts at hand-crafting ray-tracing calculations were unsuccessful. Progress accelerated after adopting the equations presented in [Laurens Neinders ' Bachelor's thesis](https://essay.utwente.nl/96153/1/Neinders_BA_EEMCS.pdf), which provided a precise geometric method for LIDAR simulation.
After extensive tuning, the system produces reasonably accurate distance measurements. As such, a new model architecture was developed that replaced the image inputs with LIDAR vectors. The reward function was also revised to include:
- Penalties for proximity to side walls.
- Positive reinforcement for maintaining balanced positioning between borders.

## [LIDAR with Gaussian Policy](/README.md):  [(lidarGauss.py)](/Scripts/Python/lidarGauss.py)

The most recent experiment in the series is built using PyTorch and a Gaussian policy. The model in this file takes simulated LIDAR data along with five telemetry inputs and produces three values representing movement decisions. These outputs are mapped to game controls: forward, backward, and turning. Compared to its predecessor (lidar.py), aside from changing the policy, this model features a larger architecture and updated reward functions.

**Experiment 7: Emulate Others**

Despite this, the model often prioritized speed-based rewards over collision penalties, leading to reckless behavior such as deliberate crashes for short-term gain.
To try something new, the model architecture was redesigned to incorporate:
- A Gaussian policy head for continuous action sampling, rather than discrete classification.
- A unified output structure with three action values, interpreted via tanh activation function (essentially sets the scale to [-1,1]). This replaced the earlier two-head system, which independently selected from {forward, backward, none} and {left, right, none}.

These changes were inspired by architectural strategies found in the [TMRL project](https://github.com/trackmania-rl/tmrl/tree/master), though my implementation uses my own approach to reward formulation. While the new design enables more expressive behavior, reward shaping remains a critical issue and continues to be a source of instability in training outcomes.

Some approaches rely on rewarding increased distance traveled along a predefined map path. However, this project intentionally avoids that method in order to prevent the model from overfitting to the predefined map path.


