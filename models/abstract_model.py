import torch.nn as nn


class AbstractBaseModel(nn.Module):
    def __init__(self, num_classes, pre_trained=False):
        super(AbstractBaseModel, self).__init__()
        self.num_classes = num_classes
        self.pre_trained = pre_trained

    def forward(self, x):
        pass
