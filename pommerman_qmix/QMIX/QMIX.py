import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import numpy.matlib
import matplotlib.pyplot as plt
from pommerman import agents
import os
import numpy as np
import pommerman
import math
from scipy import signal
#QMIXOptimizer, taken from https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py
class QMIXOptimizer(optim.Optimizer):
    """Implements QMIX-specific optimizer.
    """
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(QMIXOptimizer, self).__init__(params, defaults)

    def share_memory(self):
        pass

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                p.data.add_(-group['lr'], grad)

        return loss
