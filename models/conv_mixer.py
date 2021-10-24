import torch.nn as nn
from models.abstract_model import *
from torchvision import models


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim=1536, depth=20, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


class ConverlutionMixer(AbstractBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(ConverlutionMixer, self).__init__(num_classes, pre_trained)
        self.conver_mixer = ConvMixer(num_classes=self.num_classes)

    def forward(self, x):
        y = self.conver_mixer(x)
        return y
