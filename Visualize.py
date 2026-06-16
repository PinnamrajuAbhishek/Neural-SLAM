"""
Visualization utilities:
  1. Plot ground-truth vs. aligned predicted trajectory (3D), matches report Fig. 2.
  2. Plot per-frame ATE across the trajectory, matches report Fig. 3.
  3. Active Neural SLAM frame-sampling demo (every Nth frame), matches report sec. 6.5.

Usage:
    python visualize.py --prefix eval
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(gt_aligned, pred_aligned, save_path="trajectory_comparison.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_aligned[:, 0], gt_aligned[:, 1], gt_aligned[:, 2],
            label="Ground Truth", color="black", linewidth=2)
    ax.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
            label="Aligned Prediction", color="green", linestyle="-.")
    ax.set_title("Ground Truth vs. Aligned Predicted Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_per_frame_ate(per_frame_error, save_path="ate_per_frame.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(per_frame_error, color="purple", label="ATE per frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Error (m)")
    plt.title("Absolute Trajectory Error (per Frame)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def active_frame_sample(rgb_files, depth_files, stride=10):
    """Selects every `stride`-th frame, simulating Active Neural SLAM exploration."""
    indices = list(range(0, len(rgb_files), stride))
    return [rgb_files[i] for i in indices], [depth_files[i] for i in indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="eval",
                         help="Prefix used when saving arrays in evaluate.py")
    args = parser.parse_args()

    per_frame_error = np.load(args.prefix + "_per_frame_error.npy")
    gt_aligned = np.load(args.prefix + "_gt_aligned.npy")
    pred_aligned = np.load(args.prefix + "_pred_aligned.npy")

    plot_trajectories(gt_aligned, pred_aligned)
    plot_per_frame_ate(per_frame_error)
