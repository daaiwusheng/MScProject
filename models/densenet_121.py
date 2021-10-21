from models.abstract_model import *
from torchvision import models

class DenseNet121(AbstractBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(DenseNet121, self).__init__(num_classes, pre_trained)
        self.densenet_121 = models.densenet121(pretrained=self.pre_trained)
        self.densenet_121.classifier = nn.Linear(self.densenet_121.classifier.in_features, self.num_classes)

    def forward(self, x):
        y = self.densenet_121(x)
        return y