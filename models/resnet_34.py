from models.resnet_model import *


class ResNet34(ResNestBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(ResNet34, self).__init__(num_classes, pre_trained)
        self.resnet_34 = models.resnet34(pretrained=self.pre_trained)
        self.resnet_34.fc = nn.Linear(self.resnet_34.fc.in_features, self.num_classes)

    def forward(self, x):
        y = self.resnet_34(x)
        return y
