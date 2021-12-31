import torch.nn as nn, torch
from networks.resnet import *


class MotionEncoder(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.resnet18 = bigger_resnet18(
            kwargs

        )

    def forward(self, x):

        x = x.permute(0, 2, 1, 3, 4)
        x = self.resnet18(x)
        return x