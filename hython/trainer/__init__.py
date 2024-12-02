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
        self.epoch_preds = None 
        self.epoch_targets = None 
        self.epoch_valid_masks = None 

    def temporal_index(self, args):
        pass


    def _concat_epoch(self, pred, target, mask = None):
        if self.epoch_preds is None:
            self.epoch_preds = pred.detach().cpu().numpy()
            self.epoch_targets = target.detach().cpu().numpy()
            if mask is not None:
                self.epoch_valid_masks = mask.detach().cpu().numpy()
        else:
            self.epoch_preds = np.concatenate(
                (self.epoch_preds, pred.detach().cpu().numpy()), axis=0
            )
            self.epoch_targets = np.concatenate(
                (self.epoch_targets, target.detach().cpu().numpy()), axis=0
            )
            if mask is not None:
                self.epoch_valid_masks = np.concatenate(
                    (self.epoch_valid_masks, mask.detach().cpu().numpy()), axis=0
                )

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
