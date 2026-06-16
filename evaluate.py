"""
Evaluation: runs the trained model on the test set, aligns predicted vs.
ground-truth trajectories with Procrustes analysis, and computes Absolute
Trajectory Error (ATE).

Usage:
    python evaluate.py --data_root /path/to/data --checkpoint pose_regression_model.pth
"""

import argparse
import numpy as np
import torch
from scipy.spatial import procrustes
from torch.utils.data import DataLoader

from dataset import MultiSceneNeuralSLAMDataset, get_scene_splits
from model import PoseRegressionModel
from train import get_transforms


def compute_ate(gt_positions, pred_positions):
    """Procrustes-aligned Absolute Trajectory Error (mean Euclidean error, meters)."""
    gt = gt_positions - gt_positions.mean(axis=0)
    pred = pred_positions - pred_positions.mean(axis=0)

    mtx1, mtx2, _ = procrustes(gt, pred)
    aligned_error = np.linalg.norm(mtx1 - mtx2, axis=1)
    return aligned_error, mtx1, mtx2


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_paths = get_scene_splits(args.data_root, train_ratio=args.train_ratio)
    rgb_transform, depth_transform = get_transforms()
    test_dataset = MultiSceneNeuralSLAMDataset(test_paths, rgb_transform, depth_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = PoseRegressionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    predictions, ground_truths = [], []
    with torch.no_grad():
        for rgb, depth, pose in test_loader:
            rgb, depth = rgb.to(device), depth.to(device)
            output = model(rgb, depth).cpu().numpy()
            predictions.append(output)
            ground_truths.append(pose.numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    per_frame_error, gt_aligned, pred_aligned = compute_ate(
        ground_truths[:, :3], predictions[:, :3]
    )

    ate_rmse = np.sqrt((per_frame_error ** 2).mean())
    print(f"Test samples: {len(test_dataset)}")
    print(f"Absolute Trajectory Error (ATE, aligned): {ate_rmse:.4f} meters")

    np.save(args.output_prefix + "_per_frame_error.npy", per_frame_error)
    np.save(args.output_prefix + "_gt_aligned.npy", gt_aligned)
    np.save(args.output_prefix + "_pred_aligned.npy", pred_aligned)
    print(f"Saved arrays with prefix '{args.output_prefix}_' for plotting (see visualize.py)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="pose_regression_model.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--output_prefix", type=str, default="eval")
    args = parser.parse_args()

    evaluate(args)
