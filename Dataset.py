"""
Dataset loader for Neural SLAM (RGB-D + odometry).

Expects each scene folder to look like:

    office1/
        rgb/frame_00000.png ...
        depth/depth_00000.png ...
        odom_data.csv   (columns: x, y, z, qx, qy, qz, qw)

Multiple scene folders are merged into a single flat dataset.
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiSceneNeuralSLAMDataset(Dataset):
    """Loads RGB + Depth frame pairs along with their 6-DoF ground-truth pose."""

    def __init__(self, scene_paths, rgb_transform=None, depth_transform=None):
        self.samples = []
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        for scene_path in scene_paths:
            rgb_dir = os.path.join(scene_path, "rgb")
            depth_dir = os.path.join(scene_path, "depth")
            odom_file = os.path.join(scene_path, "odom_data.csv")

            if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir) or not os.path.exists(odom_file):
                print(f"Skipping {scene_path}: missing rgb/depth folder or odom_data.csv")
                continue

            rgb_files = sorted(os.listdir(rgb_dir))
            depth_files = sorted(os.listdir(depth_dir))
            odom_data = pd.read_csv(odom_file)

            min_len = min(len(rgb_files), len(depth_files), len(odom_data))

            for i in range(min_len):
                rgb_path = os.path.join(rgb_dir, rgb_files[i])
                depth_path = os.path.join(depth_dir, depth_files[i])
                pose = odom_data.iloc[i].values.astype(np.float32)  # [x, y, z, qx, qy, qz, qw]
                self.samples.append((rgb_path, depth_path, pose))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, pose = self.samples[idx]

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        pose = torch.tensor(pose, dtype=torch.float32)
        return rgb, depth, pose


def get_scene_splits(root_path, train_ratio=0.8, seed=42):
    """Finds all `office*` scene folders under root_path and splits them train/test."""
    import glob
    import random

    all_paths = sorted(glob.glob(os.path.join(root_path, "office*")))
    random.Random(seed).shuffle(all_paths)

    split_idx = int(train_ratio * len(all_paths))
    return all_paths[:split_idx], all_paths[split_idx:]
