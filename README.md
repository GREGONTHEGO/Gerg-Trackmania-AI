# Trackmania Reinforcement Learning Variations

This repository showcases a series of reinforcement learning experiments and model architectures developed to control a car in Trackmania using computer vision and game telemetry. These models leverage either screen captures or simulated LIDAR data, combined with in-game telemetry, to learn driving behaviors through reward-based optimization.

## Overview of the main model files
Detailed descriptions and inline documentation are available in each file under /scripts/python/.
Additional insights, challenges faced, and example videos can be found in gerg-thoughts.md.

The Primary files are listed below, ordered from most recent to earliest in developement:

1. **[LIDAR with Gaussian Policy](/EXPERIMENTS.md#lidargausspy) [(lidarGauss.py)](/Scripts/Python/lidarGauss.py)**
The most recent model in the series is built using PyTorch and a Gaussian policy. The model in this file takes simulated LIDAR data along with five telemetry inputs and produces three values representing movement decisions. Compared to its predecessor (lidar.py), aside from changing the policy, this model features a larger architecture and updated reward functions.

2. **[LIDAR with Softmax Policy](/EXPERIMENTS.md#lidarpy) [(lidar.py)](/Scripts/Python/lidar.py)**
A PyTorch model that uses discrete softmax outputs for the two sets of three values represented by, movement and turning. This served as a rough model for the Gaussian version.

3. **[CNN with PyTorch](/EXPERIMENTS.md#cnntorchpy) [(cnnTorch.py)](/Scripts/Python/cnnTorch.py)**
A PyTorch convolutional nueral network that uses combines grayscale screenshots and telemetry data to predict actions.

4. **[CNN with TensorFlow](/EXPERIMENTS.md#cnnpy) [(cnn.py)](/Scripts/Python/cnn.py)**
A TensorFlow model that uses a ConvLSTM2D (ConvLSTM2D not in pytorch) Although it works it would only run on the CPU because of the CUDA version not being supported by TensorFlow.

5. **[Basic DNN](/EXPERIMENTS.md#gamestateonlypy) [(gameStateOnly.py)](/Scripts/Python/gameStateOnly.py)**
A simple deep neural network built with TensorFlow that relies solely on telemetry inputs, without any visual information.

## Requirements for install

To install required dependencies:

```bash
pip install -r requirements.txt

```

## Acknowledgement

The distance estimation equations used are based on [Section 13 of Laurens Neinders' bachelor's thesis](https://essay.utwente.nl/96153/1/Neinders_BA_EEMCS.pdf)

Gaussian policy modeling and architectural inspiration in lidar-gauss.py were influenced by the [TMRL project](https://github.com/trackmania-rl/tmrl/tree/master)
