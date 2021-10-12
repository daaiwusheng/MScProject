
import os
import math
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from tqdm import tqdm

from . import models as nnmodels
from . import netlosses as nloss

from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import graphic as gph
from pytvision import netlearningrate
from pytvision import utils as pytutils

#----------------------------------------------------------------------------------------------
# Neural Net for Attention

class ProjectNeuralNetAbstract(NeuralNetAbstract):
    """
    Neural Net Abstract including training, validating, saving, loading, logging
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        """
        Initialization
            -patchproject (str): path project
            -nameproject (str):  name project
            -no_cuda (bool): system cuda (default is True)
            -parallel (bool)
            -seed (int)
            -print_freq (int)
            -gpu (int)
            -view_freq (in epochs)
        """

        super(ProjectNeuralNetAbstract, self).__init__(patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu)
        self.view_freq = view_freq


    def create(self,
        arch,
        num_output_channels,
        num_input_channels,
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        backbone='preactresnet'
        ):
        """
        Create
            -arch (string): architecture
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) :
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
        """


        cfg_opt={ 'momentum':momentum, 'weight_decay':weight_decay }
        #cfg_scheduler={ 'step_size':100, 'gamma':0.1  }
        cfg_scheduler={ 'mode':'min', 'patience':10  }

        self.num_classes = num_classes

        super(ProjectNeuralNetAbstract, self).create(
            arch,
            num_output_channels,
            num_input_channels,
            loss,
            lr,
            optimizer,
            lrsch,
            pretrained,
            cfg_opt=cfg_opt,
            cfg_scheduler=cfg_scheduler,
        )

        self.size_input = size_input
        self.backbone = backbone

        self.accuracy = nloss.Accuracy()

        # Set the graphic visualization
        self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )


    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:
                    x_img, y_lab = sample['image'], sample['label'].argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                x_img = x_img.cuda() if self.cuda else x_img
                y_lab_hat = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())

        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        return Y_labs, Y_lab_hats

    def __call__(self, image ):
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image
            y_lab_hat = self.net(x)
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )
        return y_lab_hat

    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained):
        """
        Create model
            -arch (string)
            -num_classes (int)
            -num_channels (int)
            -pretrained (bool)
        """

        self.net = None

        #--------------------------------------------------------------------------------------------
        # select architecture
        #--------------------------------------------------------------------------------------------
        kw = {'dim': num_output_channels, 'num_classes': self.num_classes, 'num_channels': num_input_channels, 'pretrained': pretrained}
        self.net = net_obj
        self.s_arch = arch

        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))


    def save(self, epoch, prec, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save model
        """
        print('>> save model epoch {} ({}) in {}'.format(epoch, prec, filename))
        net = self.net.module if self.parallel else self.net
        pytutils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'imsize': self.size_input,
                'num_output_channels': self.num_output_channels,
                'num_input_channels': self.num_input_channels,
                'num_classes': self.num_classes,
                'state_dict': net.state_dict(),
                'prec': prec,
                'optimizer' : self.optimizer.state_dict(),
            },
            is_best,
            self.pathmodels,
            filename
            )

    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                self.num_classes = checkpoint['num_classes']
                self._create_model(checkpoint['arch'], checkpoint['num_output_channels'], checkpoint['num_input_channels'], False )
                self.size_input = checkpoint['imsize']
                self.net.load_state_dict( checkpoint['state_dict'] )
                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True
            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))
        return bload



class ProjectNeuralNet(ProjectNeuralNetAbstract):
    """
    Attention Neural Net
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
    """
    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        super(ProjectNeuralNet, self).__init__(patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq)


    def create(self,
        arch,
        num_output_channels,
        num_input_channels,
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        backbone='preactresnet'
        ):
        """
        Create
        Args:
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) :
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """
        super(ProjectNeuralNet, self).create(
            arch,
            num_output_channels,
            num_input_channels,
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,
            backbone
        )

        self.logger_train = Logger( 'Train', ['loss', 'loss_bce', 'loss_att' ], [ 'topk'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_bce', 'loss_att' ], [ 'topk'], self.plotter )


    def training(self, data_loader, epoch=0):

        #reset logger
        self.logger_train.reset()
        data_time  = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x_org, x_img, y_mask, meta ) in enumerate(data_loader):

            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x_img.shape[0]

            y_lab = meta[:,0]
            y_theta   = meta[:,1:].view(-1, 2, 3)

            if self.cuda:
                x_org   = x_org.cuda()
                x_img   = x_img.cuda()
                y_mask  = y_mask.cuda()
                y_lab   = y_lab.cuda()
                y_theta = y_theta.cuda()

            # fit (forward)
            y_lab_hat, att, _, _ = self.net( x_img, x_org )

            # measure accuracy and record loss
            loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )
            loss_att  = self.criterion_att( x_org, y_mask, att )
            loss      = loss_bce + loss_att
            topk      = self.topk( y_lab_hat, y_lab.long() )

            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()

            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                {'topk': topk[0][0].cpu() },
                batch_size,
                )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )

    def evaluate(self, data_loader, epoch=0):

        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (x_org, x_img, y_mask, meta) in enumerate(data_loader):

                # get data (image, label)
                batch_size = x_img.shape[0]

                y_lab = meta[:,0]

                if self.cuda:
                    x_org   = x_org.cuda()
                    x_img   = x_img.cuda()
                    y_mask  = y_mask.cuda()
                    y_lab   = y_lab.cuda()


                # fit (forward)
                y_lab_hat = self.net( x_img, x_org )

                # measure accuracy and record loss
                loss  = self.criterion_bce(  y_lab_hat, y_lab.long() )

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                    {'loss': loss.cpu().item()},
                    {},
                    batch_size,
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader),
                        batch_time,
                        bplotter=False,
                        bavg=True,
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader),
            batch_time,
            bplotter=True,
            bavg=True,
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:
            self.visheatmap.show('Image', x_img.data.cpu()[0].numpy()[0,:,:])
            # self.visheatmap.show('Feature Map',srf.cpu().numpy().astype(np.float32) )


        return acc

    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:
                    x_img, y_lab = sample['image'], sample['label']
                    y_lab = y_lab.argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                if self.cuda:
                    x_img = x_img.cuda()
                y_lab_hat = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())

        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        return Y_labs, Y_lab_hats


    def __call__(self, image ):
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image
            y_lab_hat = self.net(x)
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )
        return y_lab_hat


    def _create_loss(self, loss):
        self.criterion_bce = nn.CrossEntropyLoss().cuda()
        self.s_loss = loss