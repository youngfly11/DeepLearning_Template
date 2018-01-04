import torch
import torch.nn as nn

import os
import shutil
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data, gain=np.sqrt(2.0))


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar', dir=None):

    if not os.path.exists(dir):
        os.makedirs(dir)

    # every ten epoch to save a checkpoint
    torch.save(state, os.path.join(dir, 'latest.pth.tar'))

    if (epoch) // 10 == 0 or is_best:
        torch.save(state, os.path.join(dir, filename))
        shutil.copyfile(os.path.join(dir, filename),
                        os.path.join(dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = lr*(0.95)**epoch

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__=='__main__':

    pass
