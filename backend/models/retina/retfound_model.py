import torch
import torch.nn as nn


class SimpleRETFound(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def vit_large_patch16(img_size=224, num_classes=4, **kwargs):
    return SimpleRETFound(num_classes=num_classes)