"""
Pose regression model: ResNet18 dual-encoder (RGB + Depth) -> 7D pose.

Output: [x, y, z, qx, qy, qz, qw]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseRegressionModel(nn.Module):
    def __init__(self):
        super(PoseRegressionModel, self).__init__()

        resnet_rgb = models.resnet18(pretrained=True)
        self.rgb_encoder = nn.Sequential(*list(resnet_rgb.children())[:-1])  # drop final FC

        resnet_depth = models.resnet18(pretrained=True)
        self.depth_encoder = nn.Sequential(*list(resnet_depth.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # 3 position + 4 quaternion
        )

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb).view(rgb.size(0), -1)
        depth_feat = self.depth_encoder(depth).view(depth.size(0), -1)
        fused = torch.cat((rgb_feat, depth_feat), dim=1)
        out = self.fc(fused)
        return out


def pose_loss(pred, target):
    """MSE on position + MSE on normalized quaternion."""
    pos_pred, ori_pred = pred[:, :3], pred[:, 3:7]
    pos_gt, ori_gt = target[:, :3], target[:, 3:7]

    ori_pred = ori_pred / ori_pred.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    ori_gt = ori_gt / ori_gt.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)

    pos_loss = nn.functional.mse_loss(pos_pred, pos_gt)
    ori_loss = nn.functional.mse_loss(ori_pred, ori_gt)
    return pos_loss + ori_loss
