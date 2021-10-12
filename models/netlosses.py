import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Accuracy(nn.Module):

    def __init__(self, bback_ignore=True):
        super(Accuracy, self).__init__()
        self.bback_ignore = bback_ignore

    def forward(self, y_pred, y_true):
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)

        prob = F.softmax(y_pred, dim=1).data
        prediction = torch.argmax(prob, 1)

        accs = []
        for c in range(int(self.bback_ignore), ch):
            yt_c = y_true[:, c, ...]
            num = (((prediction.eq(c) + yt_c.data.eq(1)).eq(2)).float().sum() + 1)
            den = (yt_c.data.eq(1).float().sum() + 1)
            acc = (num / den) * 100
            accs.append(acc)

        accs = torch.stack(accs)
        return accs.mean()


def centercrop(image, w, h):
    nt, ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0: image = image[:, :, padh:-padh, padw:-padw]
    return image


def flatten(x):
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat


def to_gpu(x, cuda):
    return x.cuda() if cuda else x


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]  # [N,D]
