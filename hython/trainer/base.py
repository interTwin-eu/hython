import numpy as np
import torch

from abc import ABC


from hython.utils import get_optimizer, get_lr_scheduler, generate_run_folder
from hython.metrics import MetricCollection


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

    def _set_target_weights(self):
        if self.cfg.target_weights is None or self.cfg.target_weights == "even":
            self.target_weights = {
                t: 1 / len(self.cfg.target_variables) for t in self.cfg.target_variables
            }
        else:
            raise NotImplementedError

    def _set_regularization(self):
        self.add_regularization = {}

        # return a dictionary of { reg_func1: weight1, reg_func2: weight2, ...   }

        # actually regularization should access any data in the trainig loop not only pred, target

    def _compute_batch_loss(
        self, prediction, target, valid_mask, target_weight, add_losses={}
    ):
        # Compute targets weighted loss. In case only one target, weight is 1
        loss = 0
        for i, target_name in enumerate(target_weight):
            iypred = prediction[:, i]
            iytrue = target[:, i]
            if valid_mask is not None:
                imask = valid_mask[:, i]
                iypred = iypred[imask]
                iytrue = iytrue[imask]

            w = target_weight[target_name]

            loss += w * self.cfg.loss_fn(iytrue, iypred)

        # in case there are missing observations in the batch
        # the loss should be weighted to reduce the importance
        # of the loss on the update of the NN weights
        if valid_mask is not None:
            scaling_factor = sum(valid_mask) / len(
                target
            )  # scale by number of valid samples in a mini-batch

            loss = loss * scaling_factor

        self._set_regularization()

        # compound more losses, in case dict is not empty
        for i, (reg_func, reg_weight) in enumerate(self.add_regularization):
            loss += reg_func(prediction, target) * reg_weight

        return loss

    def _backprop_loss(self, loss, opt):
        if opt is not None:
            opt.zero_grad()
            loss.backward()

            if self.cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), **self.cfg.gradient_clip
                )

            opt.step()

    def _concatenate_result(self, pred, target, mask=None):
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
        if isinstance(self.cfg.metric_fn, MetricCollection):
            metric = self.cfg.metric_fn(
                self.epoch_targets,
                self.epoch_preds,
                self.cfg.target_variables,
                self.epoch_valid_masks,
            )
        else:
            metric_or = self.cfg.metric_fn(  # {var: metric}
                self.epoch_targets,
                self.epoch_preds,
                self.cfg.target_variables,
                self.epoch_valid_masks,
            )
            metric = {}
            for itarget in metric_or:
                metric[itarget] = {
                    self.cfg.metric_fn.__class__.__name__: metric_or[itarget]
                }

        return metric

    def _get_optimizer(self):
        return get_optimizer(self.model, self.cfg)

    def _get_lr_scheduler(self, optimizer):
        return get_lr_scheduler(optimizer, self.cfg)

    def init_trainer(self, model):
        self.model = model

        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        self._set_target_weights()

        self.run_dir = generate_run_folder(self.cfg)

    def train_valid_epoch(self, model, train_loader, val_loader, device):
        model.train()

        # set time indices for training
        # TODO: This has effect only if the trainer overload the method (i.e. for RNN)
        self._set_dynamic_temporal_downsampling([train_loader, val_loader])

        train_loss, train_metric = self.epoch_step(
            model, train_loader, device, opt=self.optimizer
        )

        model.eval()
        with torch.no_grad():
            # set time indices for validation
            # This has effect only if the trainer overload the method (i.e. for RNN)
            self._set_dynamic_temporal_downsampling([train_loader, val_loader])

            val_loss, val_metric = self.epoch_step(model, val_loader, device, opt=None)

        return train_loss, train_metric, val_loss, val_metric

    def train_epoch(self):
        pass

    def valid_epoch(self):
        pass

    def epoch_step(self):
        pass

    def predict_step(self, arr, steps=-1):
        """Return the n steps that should be predicted"""
        if steps == "all":
            return arr
        elif steps == 0:
            return arr[:, -1]
        else:
            return arr[:, steps]

    def save_weights(self, model, fp=None, onnx=False):
        if fp is None:
            fp = f"{self.run_dir}/model.pt"

        if onnx:
            raise NotImplementedError()
        else:
            print(f"save weights to: {fp}")
            torch.save(model.state_dict(), fp)
