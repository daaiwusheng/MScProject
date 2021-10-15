from models.resnet_18 import *


class ModelFactory(object):
    def __init__(self):
        super(ModelFactory, self).__init__()

    def __call__(self, arch, num_classes, pre_trained):
        if arch == 'resnet18':
            return ResNet18(num_classes, pre_trained)
        else:
            assert False
