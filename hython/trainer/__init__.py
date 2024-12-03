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

    def _set_dynamic_temporal_downsampling(self, data_loaders=None, opt=None):
        """Return the temporal indices of the timeseries, it may be a subset"""

        if self.cfg.temporal_downsampling:
            if len(self.cfg.temporal_subset) > 1:
                # use different time indices for training and validation

                if opt is None:
                    # validation
                    time_range = next(iter(data_loaders[-1]))["xd"].shape[1]
                    temporal_subset = self.cfg.temporal_subset[-1]
                else:
                    time_range = next(iter(data_loaders[0]))["xd"].shape[1]
                    temporal_subset = self.cfg.temporal_subset[0]

                self.time_index = np.random.randint(
                    0, time_range - self.cfg.seq_length, temporal_subset
                )
            else:
                # use same time indices for training and validation, time indices are from train_loader
                time_range = next(iter(data_loaders[0]))["xd"].shape[1]
                self.time_index = np.random.randint(
                    0, time_range - self.cfg.seq_length, self.cfg.temporal_subset[-1]
                )

        else:
            if opt is None:
                # validation
                time_range = next(iter(data_loaders[-1]))["xd"].shape[1]
            else:
                time_range = next(iter(data_loaders[0]))["xd"].shape[1]

            self.time_index = np.arange(0, time_range)

    def _compute_loss(self, output, target, valid_mask, target_weight,  scaling_factor =None, add_losses={}):
        
        loss = self.cfg.loss_fn(
            target,
            output,
            valid_mask=valid_mask,
            scaling_factor=scaling_factor,
            target_weight=target_weight,
        )

        # compound more losses, in case dict is not empty
        # TODO: add user-defined weights
        for k in add_losses:
            loss += add_losses[k]

        return loss

    def _backprop_loss(self, loss, opt):
        if opt is not None:
            opt.zero_grad()
            loss.backward()

            if self.cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.cfg.gradient_clip)

            opt.step() 

    def _concatenate_result(self, pred, target, mask = None):
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

    def _compute_metric(self):
        metric = self.cfg.metric_fn(
            self.epoch_targets,
            self.epoch_preds,
            self.cfg.target_variables,
            self.epoch_valid_masks
            )
        return metric

    def train_valid_epoch(self, model, train_loader, val_loader, optimizer, device):
        model.train()

        # set time indices for training
        # TODO: This has effect only if the trainer overload the method (i.e. for RNN)
        self._set_dynamic_temporal_downsampling([train_loader, val_loader])

        train_loss, train_metric = self.epoch_step( # change to train_valid epoch
            model, train_loader, device, opt=optimizer
        )

        model.eval()
        with torch.no_grad():
            # set time indices for validation
            # This has effect only if the trainer overload the method (i.e. for RNN)
            self._set_dynamic_temporal_downsampling([train_loader, val_loader])

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


from .train import train_val
from .rnn import *
from .conv import *
from .cal import *
