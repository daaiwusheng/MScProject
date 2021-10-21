from models.abstract_model import *
from torchvision import models


class VGG19(AbstractBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(VGG19, self).__init__(num_classes, pre_trained)
        self.vgg_19 = models.vgg19(pretrained=self.pre_trained)
        # print(self.vgg_19)
        self.vgg_19.classifier[6] = nn.Linear(self.vgg_19.classifier[6].in_features, self.num_classes)
        # self.vgg_19.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.vgg_19.classifier = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        # )
    def forward(self, x):
        y = self.vgg_19(x)
        return y
