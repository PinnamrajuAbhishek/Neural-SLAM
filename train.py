"""
Training loop for the Neural SLAM pose regression model.

Usage:
    python train.py --data_root /path/to/data --epochs 50
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MultiSceneNeuralSLAMDataset, get_scene_splits
from model import PoseRegressionModel, pose_loss


def get_transforms():
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return rgb_transform, depth_transform


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_paths, test_paths = get_scene_splits(args.data_root, train_ratio=args.train_ratio)
    print(f"Train scenes: {len(train_paths)} | Test scenes: {len(test_paths)}")

    rgb_transform, depth_transform = get_transforms()

    train_dataset = MultiSceneNeuralSLAMDataset(train_paths, rgb_transform, depth_transform)
    test_dataset = MultiSceneNeuralSLAMDataset(test_paths, rgb_transform, depth_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    model = PoseRegressionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (rgb, depth, pose) in enumerate(train_loader):
            rgb, depth, pose = rgb.to(device), depth.to(device), pose.to(device)

            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = pose_loss(outputs, pose)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                         help="Root folder containing office1, office2, ... scene subfolders")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--save_path", type=str, default="pose_regression_model.pth")
    args = parser.parse_args()

    train(args)
