# Project Log: Reinforcement Learning in Trackmania

This document roughly outlines the five major developments of my reinforcement learning system designed to control a vehicle in Trackmania 2020. Each subsection reflects iterative improvements, experiments, and lessons learned across areas including telemetry extraction, model design, reward engineering, and architectural changes.

## [Basic DNN](/README.md):  [(gameStateOnly.py)](/Scripts/Python/gameStateOnly.py)

A simple deep neural network built with TensorFlow that relies solely on telemetry inputs, without any visual information. It uses values such as speed, position, velocity, and checkpoint count to determine the next action. While limited in perception compared to image-based models, it serves as a useful baseline for understanding the basics of a reinforcement learning model in a game environment.

**Experiment 1: Retrieving Game Data and Controlling the Vehicle**

The initial goal was to extract real-time game data from Trackmania 2020 and use that data to control the vehicle using Python. This was accomplished through Openplanet, a modding framework that enables custom plugin developement for Trackmania using AngelScript.
To implement the plugin, I wrote an 'info.toml' and a script located in 'plugins/GergBot/'. This plugin adds a menu item in Openplanet that launches a local socket server. The socket transmits the car's current speed, position, velocity and checkpoint count on each update. However, there were two issues identified early:
1. No frame-rate control: Data was sent as fast as possible, leading to inconsistencies in synchronization and strange reward behaviors.
2. TOML structuring confusion: I initially misused the 'meta' and 'script' sections, which are crucial for proper plugin registration.

On the python side, I created a function that runs two threads:
1. One that listens to the socket and updates a global 'latest_state' variable
2. The other makes movement decisions, such as, if speed is below 50 m/s, it presses the gas otherwise, it lets off.

During this phase, I also discovered that 'pynput' holds keys down continuously until explicitly released, which caused unwanted keypress spam when switching windows.

**Experiment 2: Building the first Neural Network**

https://github.com/user-attachments/assets/5507724a-d6ce-42fa-a510-b82c61f4d2be

This stage focused on building the machine learning infrastructure. I restructured the Python codebase by adding:

- genModel(): A function to generate a neural network with 8 inputs, two hidden layers of 1024 neurons, and 4 outputs (forward, backward, left, and right)
- compute_reward(): speed based reward
- policy_loss(): basic TensorFlow policy
- train(): train a given model on data from inference
- inference(): updated to call the function in real time and store the information

The inference function would gather state-action pairs, then train after a batch of runs. The first version succeeded in moving the car; however, due to a lack of sychronization between the plugin and Python threads, state sequences were mismatched. This mismatch has been fixed in more recent versions. (The fix was to only take the most recent state each time the connection recieved 1024 bytes) The model, at one point, mistakenly learned that launching off a ramp (to achieve high speed briefly) was the optimal behavior.


**Experiment 3: Teaching the model to Drive Straight**

The objective here was to get the model to learn basic straight-line driving. This turned out to be more difficult than expected due to reward misinterpretation:

- Penalizing deceleration caused the model to reverse and get stuck.
- Raw inputs with large position values resulted in spinning. (did not initialize the weights systematically)
- The model occasionally turned sharply and became stuck due to a lack of penalty in the lateral movement.

The reward function was refined to:

- Provide positive reinforcement for speed.
- Add a large bonus if a full episode performed better than prior ones.

Eventually, the model began to drive further and more consistently. A video demonstration is linked above. The speed value directly from Trackmania was positive when going forward or backward. However, if the velocity, m/s in (x,y,z) is dot product with aim direction (x,y,z) then a positive speed is given when going in the aim direction and negative when going against it. As stated in later sections, the model likes to learn that pressing only forward is the best way to complete these maps and not try to learn how to actually turn. However, it would always reach the point of only pressing forward and not actually learning when to turn.

## [CNN with TensorFlow](/README.md):  [(cnn.py)](/Scripts/Python/cnn.py)

A TensorFlow model that uses a ConvLSTM2D (ConvLSTM2D not in pytorch). Although it works, it would only run on the CPU because of my CUDA version not being supported by TensorFlow. Several hours were spent trying alternatives to make TensorFlow but those that did work would not be able to connect to the local socket server.

**Experiment 4: Integrating Convolutional Neural Networks (CNNs)**

[![YouTube Video](https://img.youtube.com/vi/-kLVGGpw-KU/0.jpg)](https://youtube.com/watch?v=-kLVGGpw-KU)

This state introduced image-based perception and transitioned the project from TensorFlow to PyTorch for improved performance and GPU compatibility.
For visual input, the DXCam library was used to capture a grayscale subseciton of the game screen in real time. Images were resized to 200x100 and stacked as sequences of recent frames, providing temporal context. Initially, 10-frame stacks were used, but this was later reduced to 5  frames to optimize relevance and memory usage.

## [CNN with PyTorch](/README.md):  [(cnnTorch.py)](/Scripts/Python/cnnTorch.py)

A PyTorch convolutional nueral network that combines grayscale screenshots and telemetry data to predict actions. Unlike the TensorFlow model this has no LSTM or 3D CNNs to gather temporal information from the stack of images that it is given. They are given as if they are different layers of the same image.

**Experiment 5: Switching libraries**

Due to limitations in TensorFlow's GPU integration on the development hardware and the absence of a PyTorch-native ConvLSTM2D, the project shifted toward alternative solutions such as 3D convolutions and LSTMs followed by 2D convolutions. This should have allowed the model to learn spatiotemporal patterns from the visual inputs. However, the 3D convolutions needed large amounts of compute and the LSTM was not letting the model learn so these were removed for just 2D convolutions.
Once PyTorch was correctly installed with CUDA support, training speed increased significantly, dropping from ~5 minutes per episode to approximately 5 seconds. This allowed for faster experimentation and tuning as well as faster inference.

To improve learning efficiency:
- The code base was modified to store multiple runs per training epoch, rather than training on the single best run.
- A best-run buffer was introduced, saving the highest-performing run observed so far. This "elite" run was combined with newly collected data each epoch to reinforce previously successful strategies.

Despite these upgrades, limitations in training data diversity and screen variations posed challenges for the CNN to generalize across runs.

## [LIDAR with Softmax Policy](/README.md):  [(lidar.py)](/Scripts/Python/lidar.py)

A PyTorch experiment that uses discrete softmax outputs for the two sets of three values represented by, movement and turning. This experiment came about when I was looking into alternatives to CNN that should allow for the model to know more about the game without just giving it seemingly random pictures.

**Experiment 6: Transistion to LIDAR-Based Perception and Reward Engineering**

[![YouTube Video](https://img.youtube.com/vi/FMvDgTzFy70/0.jpg)](https://youtube.com/watch?v=FMvDgTzFy70)

Given the resource constraints for training convolutional networks effectively, this phase explored an alternative input representation using simulated LIDAR-style data. This approach draws inspiration from Trackmania reinforcement learning woeks, where rays are cast forward from the vehicle to estimate distances to surrounding walls.
Initial attempts at hand-crafting ray-tracing calculations were unsuccessful. Progress accelerated after adopting the equations presented in [Laurens Neinders Bachelors' thesis](https://essay.utwente.nl/96153/1/Neinders_BA_EEMCS.pdf), which provided a percise geometric method for LIDAR simulation.
With a lot of experimentation with different base values, decently accurate distance measurements are now in place. As such, a new model architecture was developed that replaced the image inputs with LIDAR vectors. The reward function was also revised to include:
- Penalties for proximity to side walls.
- Positive reinforcement for maintaining central alignment.

## [LIDAR with Gaussian Policy](/README.md):  [(lidarGauss.py)](/Scripts/Python/lidarGauss.py)

The most recent experiment in the series is built using PyTorch and a Gaussian policy. The model in this file takes simulated LIDAR data along with five telemetry inputs and produces three values representing movement decisions. The three outputs are linked to the controls of the game where the first is forward, second is backward, and third goes for turning. Compared to its predecessor (lidar.py), aside from changing the policy, this model features a larger architecture and updated reward functions.

**Experiment 7: Emulate Others**

Despite this, the model often prioritized speed-based rewards over collision penalties, leading to reckless behavior such as deliberate crashes for short-term gain.
To try something new, the model architecture was redesigned to incorperate:
- A Gaussian policy head for continuous action sampling, rather than discrete classification.
- A unified output structure with three action values, interpreted via tanh activation function (essentially sets the scale to [-1,1]). This replaced the earlier two head system, which independently selected from {forward, backward, none} and {left, right, none}.

These changes were inspired by architectural strategies found in the [TMRL project](https://github.com/trackmania-rl/tmrl/tree/master), though this implementation retains its own approach to reward formulation. While the new design enables more expressive behavior, reward shaping remains a critical issue and continues to be a source of instability in training outcomes.

I have seen that in many of the reward functions they reward for increased distance on the map. However, that requires creating a map and defining more about it than what I wanted the car to know.
