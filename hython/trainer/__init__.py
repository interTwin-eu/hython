import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from torch.nn import functional as F

import importlib.util
from abc import ABC
import copy

from hython.metrics import Metric
from hython.losses import PhysicsLossCollection

tqdm_support = True if importlib.util.find_spec("tqdm") is not None else False

if tqdm_support:
    from tqdm.auto import tqdm


class BaseTrainParams:
    pass



class AbstractTrainer(ABC):
    def __init__(self):
        pass
    def temporal_index(self, args):
        pass


    def train_epoch(self):
        pass 

    def valid_epoch(self):
        pass

    def epoch_step(self):
        pass

    def predict_step(self):
        pass

    def save_weights(self, model, fp, onnx=False):
        if onnx:
            raise NotImplementedError()
        else:
            print(f"save weights to: {fp}")
            torch.save(model.state_dict(), fp)


from .train import train_val, metric_epoch, loss_batch
from .rnn import *
from .conv import *
from .cal import *
