#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from pathlib import Path
import datetime
import shutil
import torch.nn as nn
import torch.nn.functional as F
import logging
import coloredlogs
import os
import cv2
import torch
import functools
import numpy as np
import math
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from decimal import Decimal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
from torch.autograd import Variable


def to_numpy(var):
    # return var.cpu().data.numpy()
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


# logging
def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))
    

def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


def get_logger(file_path, name='ED'):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger(name)
    coloredlogs.install(level='INFO', logger=logger)

    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def print_params(config, prtf=print):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(config.items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")


def as_markdown(config):
    """ Return configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(config.items()):
        text += "|{}|{}|  \n".format(attr, value)

    return text

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

# def distillation(criterion,outputs, labels, teacher_outputs, params):
#     """
#     Compute the knowledge-distillation (KD) loss given outputs, labels.
#     "Hyperparameters": temperature and alpha
#     NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
#     and student expects the input tensor to be log probabilities! See Issue #2
#     """
#     alpha = params.alpha
#     T = params.temperature
#     KD_loss = nn.KLDivLoss(reduction='mean')(torch.nn.functional.log_softmax(outputs/T, dim=1),
#                             torch.nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) +\
#                             criterion(outputs, labels) * (1. - alpha)
#     return KD_loss

def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum') * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def pix_loss(x,y):
    loss = torch.mean(torch.mean(torch.abs(x-y), dim = (1,2,3)))
    return loss
    
####################
# image convert
####################

def _make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def get_activation(name,activation):
    def hook(model, input, output):
        activation[name] = output
    return hook

def plot_loss(args,loss,apath,epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

def plot_psnr(args,apath,epoch,log):
    
    axis = np.linspace(1, epoch, epoch)
    for idx_data, d in enumerate(args.data_test):
        label = 'SR on {}'.format(d)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(args.scale):
            plt.plot(
                axis,
                log[:, idx_data, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(apath, 'test_{}_{}.png'.format(d, args.save)))
        plt.close(fig)

def plot_bit(args,apath,epoch,log):
    
    axis = np.linspace(1, epoch, epoch)
    for idx_data, d in enumerate(args.data_test):
        label = 'SR on {}'.format(d)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(args.scale):
            plt.plot(
                axis,
                log[:, idx_data, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Avg Bit')
        plt.grid(True)
        plt.savefig(os.path.join(apath, 'test_{}_bit_{}.png'.format(d, args.save)))
        plt.close(fig)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', lpips=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        if lpips:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_lpips.pth.tar'))
        else:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    else:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_latest.pth.tar'))
