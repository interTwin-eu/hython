from typing import Optional, List
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class RegCollection(nn.Module):
    def __init__(self, loss: List[nn.Module] = None):
        super(RegCollection, self).__init__()

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


class Reg1(nn.Module):
    __name__ = "PrecipSoilMoisture"

    def __init__(self):
        super(Reg1, self).__init__()

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


class ThetaReg(nn.Module):
    __name__ = "Theta"

    def __init__(self, min_storage=0):
        super(ThetaReg, self).__init__()
        self.min_storage = min_storage

    def forward(self, thetaS, thetaR):
        viol = F.relu(((thetaR + self.min_storage) - thetaS))
        loss = torch.sum(viol**2) / max(torch.sum(viol), 1)
        return {self.__name__: loss}



class RangeBoundReg(nn.Module):
    def __init__(self, cfg) -> None:
        super(RangeBoundReg, self).__init__()
        self.cfg = cfg
        self.lb = torch.tensor(self.cfg.lb)
        self.ub = torch.tensor(self.cfg.ub)
        self.factor = torch.tensor(self.cfg.factor)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        loss = 0
        for i in range(len(inputs)):
            lb = self.lb[i]
            ub = self.ub[i]
            upper_bound_loss = torch.relu(inputs[i] - ub)
            lower_bound_loss = torch.relu(lb - inputs[i])
            mean_loss = self.factor * (upper_bound_loss + lower_bound_loss).mean() / 2.0
            loss = loss + mean_loss
        return loss