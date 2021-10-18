from models.resnet_model import *


class ResNet50(ResNestBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(ResNet50, self).__init__(num_classes, pre_trained)
        self.resnet_50 = models.resnet50(pretrained=self.pre_trained)
        self.resnet_50.fc = nn.Linear(self.resnet_50.fc.in_features, self.num_classes)

    def forward(self, x):
        y = self.resnet_50(x)
        return y
