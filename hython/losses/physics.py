from typing import Optional, List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class PhysicsLossCollection:
    def __init__(self, loss: List[_Loss] = None):
        super(PhysicsLossCollection, self).__init__()

        if not isinstance(loss, list) and loss is not None:
            loss = [loss]

        if loss is not None:
            self.losses = nn.ModuleDict({l.__name__: l for l in loss})
        else:
            self.losses = {}

    def __getitem__(self, k):
        if k in self.losses:
            return self.losses[k]
        else:
            return return_dict


def return_dict(*args):
    return {}


class PrecipSoilMoistureLoss(_Loss):
    __name__ = "PrecipSoilMoisture"

    def __init__(self):
        super(PrecipSoilMoistureLoss, self).__init__()

    def forward(self, x, y):
        N, T, C = x.shape

        # compute the x and y deltas, and remove the first element from the time vector due to torch.roll logic
        diff_x = (x - x.roll(1, dims=1))[:, 1:]
        diff_y = (y - y.roll(1, dims=1))[:, 1:]
        # positive increments of the x field should produce positive increments of the y field
        positive_x = diff_x >= 0
        # positive
        loss = torch.sum((F.relu(-1 * diff_y[positive_x])) ** 2) / torch.sum(positive_x)

        return {self.__name__: loss}


class ThetaLoss(_Loss):
    __name__ = "Theta"

    def __init__(self, min_storage=0):
        super(ThetaLoss, self).__init__()
        self.min_storage = min_storage

    def forward(self, thetaS, thetaR):
        viol = F.relu(((thetaR + self.min_storage) - thetaS))
        loss = torch.sum(viol**2) / max(torch.sum(viol), 1)
        return {self.__name__: loss}
