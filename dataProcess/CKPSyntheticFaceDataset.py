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

from dataProcess.ckp_dataprovider import *
from dataProcess.CKPSyntheticGenerator import *

GENERATE_IMAGE = 'image'
GENERATE_IMAGE_SYN = 'image_and_mask'

class CKPSyntheticFaceDataset(data.Dataset):
    '''
     Management for Synthetic Face dataset
     '''
    # generate_image = 'image'
    # generate_image_and_mask = 'image_and_mask'

    def __init__(self,
                 is_train=True,
                 pathnameback=None,
                 ext='jpg',
                 count=None,
                 num_channels=3,
                 generate=GENERATE_IMAGE_SYN,
                 iluminate=True, angle=45, translation=0.3, warp=0.1, factor=0.2,
                 transform_image=None,
                 transform_data=None,
                 ):
        """Initialization
          """

        self.data_provider = CKPDataProvider(reconstruct=False, is_train=is_train)
        self.bbackimage = pathnameback != None
        self.databack = None

        if self.bbackimage:
            pathnameback = os.path.expanduser(pathnameback)
            self.databack = imutl.imageProvide(pathnameback, ext=ext)
        self.num_channels = num_channels
        self.generate = generate
        self.ren = Generator(iluminate, angle, translation, warp, factor)
        if count is None:
            self.count = len(self.data_provider)
        else:
            self.count = count

        self.transform_image = transform_image
        self.transform_data = transform_data

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # read image
        image, label = self.data_provider.get_data(idx)

        image = utility.to_channels(image, self.num_channels)

        # read background
        if self.bbackimage:
            idxk = random.randint(1, len(self.databack) - 1)
            back = self.databack[idxk]
            back = F.resize_image(back, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);
            back = utility.to_channels(back, self.num_channels)

            # show_image(back,'back')
        else:
            back = np.ones((640, 1024, 3), dtype=np.uint8) * 255

        if self.generate == GENERATE_IMAGE:
            obj = ObjectImageTransform(image)

        elif self.generate == GENERATE_IMAGE_SYN:

            image_org, image_ilu, mask, h = self.ren.generate(image, back)

            image_org = utility.to_gray(image_org.astype(np.uint8))
            image_org = utility.to_channels(image_org, self.num_channels)
            image_org = image_org.astype(np.uint8)

            image_ilu = utility.to_gray(image_ilu.astype(np.uint8))
            image_ilu = utility.to_channels(image_ilu, self.num_channels)
            image_ilu = image_ilu.astype(np.uint8)

            mask = mask[:, :, 0]
            mask_t = np.zeros((mask.shape[0], mask.shape[1], 2))
            mask_t[:, :, 0] = (mask == 0).astype(np.uint8)  # 0-backgraund
            mask_t[:, :, 1] = (mask == 1).astype(np.uint8)

            obj_image = ObjectImageTransform(image_ilu.copy())
            # obj_data = ObjectImageAndMaskMetadataTransform(image_ilu.copy(), mask_t,
            #                                                np.concatenate(([label], h), axis=0))

        else:
            assert (False)

        if self.transform_image:
            if self.generate == GENERATE_IMAGE:
                obj_image = self.transform_image(obj)
            else:
                obj_image = self.transform_image(obj_image)

        x_img = obj_image.to_value()
        y_lab = torch.tensor(int(label))

        # if self.transform_data:
        #     if self.generate == GENERATE_IMAGE_SYN:
        #         obj_image = self.transform_image(obj_image)
        #         x_img = obj_image.to_value()
        #         y_lab = torch.tensor(int(label))

        return x_img, y_lab
