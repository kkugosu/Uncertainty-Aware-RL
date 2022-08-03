import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.autograd.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device", device)
batch_size = 24
hidden_size = 10
state_size = 1
action_size = 1
reward_size = 1
freeze = 0

class Custom_Activation_F:

    def __init__(self, rate=1):
        self.rate = rate

    def logact(self, a):
        '''
        logistic activation function
        '''
        i = 0
        while i < len(a):
            j = 0
            while j < len(a[i]):
                if a[i][j] > 0:
                    a[i][j] = torch.log(a[i][j] + self.rate)
                else:
                    a[i][j] = - torch.log(self.rate - a[i][j])
                j = j + 1
            i = i + 1
        return a


class BayesianLinear(nn.Module):

    def __init__(self, i_s, o_s):
        '''
        i_s = input_size
        o_s = output_size

        '''
        super().__init__()
        self.i_s = i_s
        self.o_s = o_s
        self.b_s = 1
        self.w = nn.Parameter(
            torch.zeros(self.i_s, self.o_s, dtype=torch.float32, requires_grad=True)
        )
        self.b = nn.Parameter(
            torch.zeros(self.o_s, dtype=torch.float32, requires_grad=True)
        )

        self.w_prior = torch.zeros(self.i_s, self.o_s)
        self.b_prior = torch.zeros(self.o_s)

    def _rep(self, mu):
        return mu + torch.randn_like(mu) * 0.1

    def _update_prior(self, w1, w2, b, rate=0.1):
        self.w_prior = w.clone().detach() * rate + self.w_prior * (1 - rate)
        self.b_prior = b.clone().detach() * rate + self.b_prior * (1 - rate)

    def kldloss(self):
        sum1 = torch.sum(torch.square(self.w - self.w_prior))
        sum2 = torch.sum(torch.square(self.b - self.b_prior))
        return sum1 + sum2

    def forward(self, x):
        self.b_s = len(x)
        b = self._rep(self.b)
        w = self._rep(self.w)
        b = b.expand(self.b_s, self.o_s)
        x = torch.matmul(x, w) + b
        # self._update_prior(self.w1_prior, self.w2_prior, self.b_prior, rate)
        # if we want to move prior, we can just subtract _prior term at the upper line
        return x


class Bayesian_Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(0.1)
        self.blinear1 = BayesianLinear(input_size, hidden_size)
        self.blinear2 = BayesianLinear(hidden_size, hidden_size)
        self.blinear3 = BayesianLinear(hidden_size, output_size)
        self.myact = Custom_Activation_F()

    def forward(self, x):
        x = self.blinear1(x)

        x = self.tanh(x)
        x = self.blinear2(x)

        x = self.tanh(x)
        x = self.blinear3(x)

        # self._update_prior(self.w1_prior, self.w2_prior, self.b_prior, rate)
        # if we want to move prior, we can just subtract _prior term at the upper line
        return x

    def kld_loss(self):
        L1 = self.blinear1.kldloss()
        L2 = self.blinear2.kldloss()
        L3 = self.blinear3.kldloss()

        return (L1 + L2 + L3)