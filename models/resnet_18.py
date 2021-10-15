from models.resnet_model import *


class ResNet18(ResNestBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(ResNet18, self).__init__(num_classes, pre_trained)
        self.resnet_18 = models.resnet18(pretrained=self.pre_trained)
        self.resnet_18.fc = nn.Linear(self.resnet_18.fc.in_features, self.num_classes)

    def forward(self, x):
        y = self.resnet_18(x)
        return y
