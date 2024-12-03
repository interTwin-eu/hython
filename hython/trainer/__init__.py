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
        self.model = None
        self.device = None

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

    def train_valid_epoch(self, model, train_loader, val_loader, optimizer, device):
        model.train()

        # set time indices for training
        # This has effect only if the trainer overload the method (i.e. for RNN)
        self.temporal_index([train_loader, val_loader])

        train_loss, train_metric = self.epoch_step( # change to train_valid epoch
            model, train_loader, device, opt=optimizer
        )

        model.eval()
        with torch.no_grad():
            # set time indices for validation
            # This has effect only if the trainer overload the method (i.e. for RNN)
            self.temporal_index([train_loader, val_loader])

            val_loss, val_metric = self.epoch_step( # change to train_valid epoch
                model, val_loader, device, opt=None
            )
        
        return train_loss, train_metric, val_loss, val_metric

    def train_epoch(self):
        pass

    def valid_epoch(self):
        pass

    def epoch_step(self):
        pass

    def predict_step(self, arr, steps=-1):
        """Return the n steps that should be predicted"""
        return arr[:, steps]

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
