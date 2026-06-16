# Neural SLAM using RGB-D Data and Pose Regression

Deep learning-based SLAM pipeline that predicts 6-DoF camera poses directly
from RGB-D image pairs using a dual ResNet18 encoder, with a simplified
Active Neural SLAM frame-sampling strategy for efficient exploration.

**Achieved ATE: 0.0156 m** (Procrustes-aligned) on held-out indoor scenes.

Full writeup: [`docs/Neural_SLAM_Report.pdf`](docs/Neural_SLAM_Report.pdf)

## Overview

Traditional SLAM relies on handcrafted feature extraction and geometric
estimation, which struggles in visually complex or dynamic environments.
This project trains a CNN to regress 6-DoF pose (3D position + quaternion
orientation) directly from RGB and depth frames, removing the need for
manual feature engineering.

**Pipeline:**
1. RGB and depth frames are encoded separately using two ResNet18 backbones
   (ImageNet-pretrained, final FC layer removed).
2. Features are concatenated and passed through a fully connected head that
   outputs a 7D pose vector `[x, y, z, qx, qy, qz, qw]`.
3. Quaternion outputs are L2-normalized during loss computation to keep
   orientation predictions on the unit sphere.
4. Predicted trajectories are aligned to ground truth using **Procrustes
   analysis** (corrects scale/rotation/translation offsets) before computing
   **Absolute Trajectory Error (ATE)**.
5. An **Active Neural SLAM** frame-sampling strategy selects every Nth frame
   (default N=10) instead of processing every frame, simulating selective
   exploration to reduce compute.

## Repo Structure

```
neural-slam-repo/
├── src/
│   ├── dataset.py      # MultiSceneNeuralSLAMDataset + train/test scene split
│   ├── model.py         # ResNet18 dual-encoder PoseRegressionModel + loss
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Procrustes-aligned ATE computation
│   └── visualize.py      # Trajectory plots + active frame sampling demo
├── docs/
│   └── Neural_SLAM_Report.pdf
├── assets/
│   └── (sample trajectory plots go here)
├── requirements.txt
└── README.md
```

## Data Format

Each scene folder should be structured as:

```
office1/
├── rgb/frame_00000.png ...
├── depth/depth_00000.png ...
└── odom_data.csv   # columns: x, y, z, qx, qy, qz, qw
```

Place all scene folders (`office1`, `office2`, ...) under one `data_root`
directory. The dataset class in `dataset.py` was originally built around a
ROS2 rosbag → PNG/CSV extraction pipeline (RGB topic, depth topic, and
odometry messages parsed via the `rosbags` library); the extraction script
itself is not included here since it is environment-specific, but any
RGB-D + 6-DoF pose dataset in the format above will work directly with this
code.

## Usage

```bash
pip install -r requirements.txt

# Train
python src/train.py --data_root /path/to/data --epochs 50 --batch_size 8

# Evaluate (computes Procrustes-aligned ATE)
python src/evaluate.py --data_root /path/to/data --checkpoint pose_regression_model.pth

# Visualize trajectories + per-frame error
python src/visualize.py --prefix eval
```

## Results

| Metric | Value |
|---|---|
| Absolute Trajectory Error (ATE) | 0.0156 m |
| Backbone | ResNet18 (dual encoder, ImageNet-pretrained) |
| Epochs | 50 |
| Optimizer | Adam, lr=1e-4 |
| Active SLAM frame stride | every 10th frame |

Trajectory alignment was performed using Procrustes analysis (scale +
rotation correction) prior to ATE computation.

## Future Work

- Incorporate dynamic object filtering for non-static scenes
- Reinforcement-learning-based frame selection for true Active SLAM
  (current version uses a fixed stride, not a learned policy)
- Extend to full 3D semantic mapping



Built as part of M.Tech coursework (Neural Networks & Deep Learning,
AI & Data Analytics, NIT Calicut). See full report for literature review
and detailed methodology.
