import os
import numpy as np
import cv2
import random

import torch
import torch.utils.data as data

from pytvision.datasets import imageutl as imutl
from pytvision.datasets import utility
from pytvision.transforms import functional as F

from pytvision.transforms.aumentation import (
    ObjectImageMaskAndWeightTransform,
    ObjectImageTransform,
    ObjectImageAndLabelTransform,
    ObjectImageAndMaskTransform,
    ObjectRegressionTransform,
    ObjectImageAndAnnotations,
    ObjectImageAndMaskMetadataTransform,
)

from dataProcess.fer2013_provider import *


class FER2013Dataset(data.Dataset):
    def __init__(self,
                 num_channels=3,
                 image_size=112,
                 transform_image=None,
                 ):
        """Initialization
          """
        self.data_provider = FER2013Provider()
        self.num_channels = num_channels
        self.image_size = image_size
        self.count = len(self.data_provider)
        self.transform_image = transform_image

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # read image
        image, label = self.data_provider.get_data(idx)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        # show_image(image,"fer")
        image = utility.to_channels(image, self.num_channels)
        obj = ObjectImageTransform(image)
        obj_image = self.transform_image(obj)

        x_img = obj_image.to_value()
        y_lab = torch.tensor(int(label))

        return x_img, y_lab