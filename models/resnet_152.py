from models.resnet_model import *


class ResNet152(ResNestBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(ResNet152, self).__init__(num_classes, pre_trained)
        self.resnet_152 = models.resnet152(pretrained=self.pre_trained)
        self.resnet_152.fc = nn.Linear(self.resnet_152.fc.in_features, self.num_classes)

    def forward(self, x):
        y = self.resnet_152(x)
        return y