import os
import numpy as np

import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms, utils

from dataProcess.CKPSyntheticFaceDataset import *
from models.projectNet import *

from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

import datetime
from argparse import ArgumentParser
from dataProcess.aug import get_transforms_aug, get_transforms_det
from models.projectNet import *

# parameters
DATABACK = '/databig/coco/unlabeled2017/'
DATA = '~/.datasets'
NAMEDATASET = 'ckp_by_myself'
PROJECT = '/databig/projectLog'  # write log to this disk on Linux
EPOCHS = 100
TRAINITERATION = 16000
TESTITERATION = 1600
BATCHSIZE = 64  # 32, 64, 128, 160, 200, 240
LEARNING_RATE = 0.0001
MOMENTUM = 0.5
PRINT_FREQ = 50
WORKERS = 1
RESUME = 'model_best.pth.tar'  # chk000000, model_best
GPU = 0
NAMEMETHOD = 'ProjectStatisticsNeuralNet'  # ProjectNeuralNet, ProjectStatisticsNeuralNet
ARCH = 'resnet101'  # resnet18
PRE_Trained = False  # if using pretrained model, set True
GENERATER_Mode = GENERATE_IMAGE_SYN  # GENERATE_IMAGE_SYN
LOSS = 'cross_entropy_loss'
OPT = 'adam'
SCHEDULER = 'fixed'
NUMCLASS = 8  # 6, 7, 8
NUMCHANNELS = 3
DIM = 32
SNAPSHOT = 10
IMAGESIZE = 112  # according to the neural network input
KFOLD = 0
NACTOR = 10
BACKBONE = 'resnet101'  # resnet18, resnet18_pretrained,resnet34_pretrained,resnet50_pretrained
B_Train = True  # True, False for evaluate the model

EXP_NAME = 'MSc_' + NAMEMETHOD + '_' + ARCH + '_' + LOSS + '_' + OPT + '_' + NAMEDATASET + '_dim' + str(
    DIM) + '_bb_' + BACKBONE + '_Epoch_'+str(EPOCHS) + '_fold_' + str(KFOLD) + '_000'

'''
check list before run this file:
PRE_Trained
ARCH
IMAGESIZE
BACKBONE
B_Train
GENERATER_Mode
'''

# experiment name


def main():
    print('start running:', PRE_Trained )
    # parameters
    imsize = IMAGESIZE
    parallel = False
    num_classes = NUMCLASS
    num_channels = NUMCHANNELS
    dim = DIM
    view_freq = 1

    if B_Train:
        trainiteration = TRAINITERATION  #
        testiteration = TESTITERATION  #
    else:
        trainiteration = None  #
        testiteration = None  #
        # GENERATER_Mode = GENERATE_IMAGE  # if B_train is false, turn this sentence on

    no_cuda = False
    seed = 1
    finetuning = False
    balance = False
    fname = NAMEMETHOD

    net_train = {
        'ProjectNeuralNet': ProjectNeuralNet,
        'ProjectStatisticsNeuralNet': ProjectStatisticsNeuralNet
    }

    network = net_train[fname](
        patchproject=PROJECT,
        nameproject=EXP_NAME,
        no_cuda=no_cuda,
        parallel=parallel,
        seed=seed,
        print_freq=PRINT_FREQ,
        gpu=GPU,
        view_freq=view_freq,
        pre_trained=PRE_Trained
    )

    network.create(
        arch=ARCH,
        num_output_channels=DIM,
        num_input_channels=NUMCHANNELS,
        loss=LOSS,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        optimizer=OPT,
        lrsch=SCHEDULER,
        pretrained=finetuning,
        size_input=imsize,
        num_classes=num_classes
    )

    # resume
    network.resume(os.path.join(network.pathmodels, RESUME))
    cudnn.benchmark = True

    # kfold = KFOLD
    # nactores = NACTOR
    # idenselect = np.arange(nactores) + kfold * nactores

    # datasets
    # training dataset
    train_data = CKPSyntheticFaceDataset(
        is_train=True,
        pathnameback=DATABACK,
        ext='jpg',
        count=trainiteration,
        num_channels=NUMCHANNELS,
        generate=GENERATER_Mode,
        iluminate=True, angle=30, translation=0.2, warp=0.1, factor=0.2,
        transform_data=get_transforms_aug(imsize),
        transform_image=get_transforms_det(imsize),
    )

    num_train = len(train_data)
    sampler = SubsetRandomSampler(np.random.permutation(num_train))

    train_loader = DataLoader(
        train_data,
        batch_size=BATCHSIZE,
        num_workers=WORKERS,
        pin_memory=network.cuda,
        drop_last=True,
        sampler=sampler,
        # shuffle=True
    )

    # validate dataset
    # SyntheticFaceDataset, SecuencialSyntheticFaceDataset
    val_data = CKPSyntheticFaceDataset(
        is_train=False,
        pathnameback=DATABACK,
        ext='jpg',
        count=testiteration,
        num_channels=NUMCHANNELS,
        generate=GENERATER_Mode,
        iluminate=True, angle=30, translation=0.2, warp=0.1, factor=0.2,
        transform_data=get_transforms_aug(imsize),
        transform_image=get_transforms_det(imsize),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=network.cuda,
        drop_last=False
    )

    # print neural net class
    print('SEG-Torch: {}'.format(datetime.datetime.now()))
    print(network)

    # training neural net
    if B_Train:
        network.fit(train_loader, val_loader, EPOCHS, SNAPSHOT)
    else:
        network.evaluate_model(val_loader)

    print("Optimization Finished!")
    print("DONE!!!")


if __name__ == '__main__':
    main()
