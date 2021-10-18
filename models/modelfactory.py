from models.resnet_18 import *
from models.resnet_34 import *
from models.resnet_50 import *


class ModelFactory(object):
    def __init__(self):
        super(ModelFactory, self).__init__()

    def __call__(self, arch, num_classes, pre_trained):
        if arch == 'resnet18':
            return ResNet18(num_classes, pre_trained)
        elif arch == 'resnet34':
            return ResNet34(num_classes, pre_trained)
        elif arch == 'resnet50':
            return ResNet50(num_classes, pre_trained)
        else:
            assert False
