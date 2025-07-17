# Trackmania Reinforcement Learning Variations

This repository contains a series of experiments and architectures designed to control a car in Trackmania using reinforcement learning and computer vision. The models utilized game telemetry and either pictures of the game or lidar of walls. This information was used to teach the model driving behaviors through chasing better rewards.

## Overview of the main model files
For more information about the files check the comments left in the files or my gerg-thoughts.md which is a small write up of the challenges that I have faced upto this point.

1. **lidar-gauss.py**
The most recent addition to the series of tests. This uses three outputs to determine all movements. Uses LIDAR simulation and a gaussian policy head. The main changes in this file compared to the lidar.py is that this one has a larger model and a slightly different reward structure.

2. **lidar.py**
The predecessor to the gauss file above used softmax and two different outputs of three possibilities to determine movements and turning.

3. **cnnTorch.py**
A pyTorch-based convolutional nueral network that uses a series of grayscale screen scrapes and telemetry data to infer the next best move.

4. **cnn.py**
A TensorFlow model that uses an LSTMConvolution (not in pytorch) and was replaced as it took ages to run and the inference time of one frame was abyssmal

5. **Game-State-Only.py**
A tensorflow deep neural network that only uses telemetry data for the car to learn anything.

## Requirements for install

Intall dependencies with:

```bash
pip install -r requirements.txt

```

## Acknoledgement

I used the equations from section 13 of this paper: https://essay.utwente.nl/96153/1/Neinders_BA_EEMCS.pdf

Also the idea of using gaussian distribution and changing what my outputs were are from: https://github.com/trackmania-rl/tmrl/tree/master