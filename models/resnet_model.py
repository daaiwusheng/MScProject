from models.abstract_model import *
from torchvision import models


class ResNestBaseModel(AbstractBaseModel):
    def __init__(self, num_classes):
        super(ResNestBaseModel, self).__init__(num_classes)
