from models.resnet_model import *


class ResNet101(ResNestBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(ResNet101, self).__init__(num_classes, pre_trained)
        self.resnet_101 = models.resnet101(pretrained=self.pre_trained)
        self.resnet_101.fc = nn.Linear(self.resnet_101.fc.in_features, self.num_classes)

    def forward(self, x):
        y = self.resnet_101(x)
        return y