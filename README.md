# Trackmania Reinforcement Learning Variations

This repository showcases a series of reinforcement learning experiments and model architectures developed to control a car in Trackmania using computer vision and game telemetry. These models leverage either screen captures or simulated LIDAR data, combined with in-game telemetry, to learn driving behaviors through reward-based optimization.

## Overview of the main model files
Detailed descriptions and inline documentation are available in each file under /scripts/python/.
Additional insights, challenges faced, and example videos can be found in EXPERIMENTS.md.
Each of the below experiment entries links to the relevant section within EXPERIMENTS.md, and the filename links to the python project file.

The Primary experiments are listed below, ordered from most recent to earliest in development:

1. **[LIDAR with Gaussian Policy](/EXPERIMENTS.md#lidar-with-gaussian-policy--lidargausspy):  [(lidarGauss.py)](/Scripts/Python/lidarGauss.py)**
The most recent experiment in the series uses “LIDAR” with Gaussian policy and is built using PyTorch and a Gaussian policy. The output of this model is three values that are sent through the tanh function to return a real number on the scale of [-1,1].

2. **[LIDAR with Softmax Policy](/EXPERIMENTS.md#lidar-with-softmax-policy--lidarpy):  [(lidar.py)](/Scripts/Python/lidar.py)**
An experiment using “LIDAR” with a PyTorch model that uses discrete softmax outputs for the two sets of three values represented by movement and turning.

3. **[CNN with PyTorch](/EXPERIMENTS.md#cnn-with-pytorch--cnntorchpy):  [(cnnTorch.py)](/Scripts/Python/cnnTorch.py)**
A PyTorch convolutional neural network that combines grayscale screenshots and telemetry data to predict actions.

4. **[CNN with TensorFlow](/EXPERIMENTS.md#cnn-with-tensorflow--cnnpy):  [(cnn.py)](/Scripts/Python/cnn.py)**
A TensorFlow model that uses 2D Convolutional Long Short Term Memory (ConvLSTM2D) for the CNN layers.

5. **[Basic DNN](/EXPERIMENTS.md#basic-dnn--gamestatesonlypy):  [(gameStateOnly.py)](/Scripts/Python/gameStateOnly.py)**
A simple deep neural network built with TensorFlow that relies solely on telemetry inputs, without any visual information.

## Requirements for installation

To install required dependencies:

```bash
pip install -r requirements.txt

```

## Acknowledgement

The distance estimation equations used are based on [Section 13 of Laurens Neinders' bachelor's thesis](https://essay.utwente.nl/96153/1/Neinders_BA_EEMCS.pdf)

Gaussian policy modeling and architectural inspiration in lidar-gauss.py were influenced by the [TMRL project](https://github.com/trackmania-rl/tmrl/tree/master)

Full Title: Trackmania

Release Year: 2020 (specifically July 1, 2020, for Windows)

Developer: Ubisoft Nadeo

Publisher: Ubisoft
