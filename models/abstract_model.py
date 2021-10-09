import torch.nn as nn


class AbstractBaseModel(nn.Module):
    def __init__(self,num_classes):
        self.num_classes = num_classes

    def forward(self, x):
        pass
