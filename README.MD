# Neural SLAM Using RGB-D Data and Pose Regression

## Overview

This project implements a Neural SLAM (Simultaneous Localization and Mapping) system that estimates 6-DoF camera poses directly from RGB-D data using deep learning.

The system leverages a ResNet18-based pose regression network to predict translation and orientation parameters from synchronized RGB and depth images. Post-processing techniques including trajectory alignment and Gaussian smoothing are applied to improve localization accuracy.

Additionally, the project explores Active Neural SLAM concepts through selective frame sampling strategies that reduce computational overhead while preserving mapping quality.

## Key Features

* RGB-D based pose estimation
* ResNet18 pose regression architecture
* Quaternion-based orientation prediction
* Trajectory alignment using Procrustes/Kabsch methods
* Gaussian trajectory smoothing
* Active frame selection strategy
* Absolute Trajectory Error (ATE) evaluation

## Methodology

1. RGB and depth images are extracted from ROS bag datasets.
2. Images are preprocessed and resized.
3. RGB-D inputs are fed into a ResNet18-based neural network.
4. The model predicts 6-DoF poses:

   * Position (x, y, z)
   * Orientation (qx, qy, qz, qw)
5. Predicted trajectories are aligned and smoothed.
6. Performance is evaluated using Absolute Trajectory Error (ATE).

## Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Pandas
* SciPy
* Matplotlib

## Applications

* Autonomous Navigation
* Mobile Robotics
* Indoor Localization
* Intelligent Mapping Systems
* Autonomous Exploration

## Future Work

* Reinforcement Learning based Active SLAM
* Dynamic Object Handling
* Semantic Mapping
* Real-time Deployment
* Multi-Sensor Fusion
