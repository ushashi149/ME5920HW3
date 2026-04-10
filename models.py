

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18_2d(num_classes: int, pretrained: bool = False) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_r3d_18(num_classes: int, pretrained: bool = False) -> nn.Module:
    w = models.R3D_18_Weights.DEFAULT if pretrained else None
    m = models.video.r3d_18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class CNNLSTM(nn.Module):
    

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.embed_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.lstm = nn.LSTM(
            self.embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b, t, c, h, w = x.shape
        x_flat = x.view(b * t, c, h, w)
        feat = self.backbone(x_flat)
        feat = feat.view(b, t, -1)
        out, _ = self.lstm(feat)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last)

    def set_backbone_requires_grad(self, requires: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = requires
