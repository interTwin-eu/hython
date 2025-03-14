import numpy as np
import torch
import logging 
from abc import ABC
from typing import Dict, Iterable, List

from hython.utils import get_optimizer, get_lr_scheduler, get_temporal_steps, generate_run_folder
from hython.metrics import MetricCollection
from hython.models.head import *

LOGGER = logging.getLogger(__name__)

class AbstractTrainer(ABC):
    def __init__(self, cfg):
        self.epoch_preds = None
        self.epoch_targets = None
        self.epoch_valid_masks = None
        self.model = None
        self.device = None

        self.cfg = cfg

        self.run_dir = generate_run_folder(self.cfg)

        LOGGER.info(f"Run directory: {self.run_dir}")         

    def _set_dynamic_temporal_downsampling(self, data_loaders=None, opt=None):
        """Return the temporal indices of the timeseries, it may be a subset"""
        
        try:
            temporal_downsampling = self.cfg.temporal_downsampling
        except:
            temporal_downsampling = False
            
        if temporal_downsampling:
            if len(self.cfg.temporal_subset) > 1:
                # use different time indices for training and validation
                if opt is None:
                    # validation
                    time_range = next(iter(data_loaders[-1]))["xd"].shape[1]
                    temporal_subset_size = self.cfg.temporal_subset[-1]

                    avail_time = (time_range - self.cfg.seq_length) - temporal_subset_size
                    if avail_time > 0:
                        choice = np.arange(0, time_range - self.cfg.seq_length, 1)
                        self.time_index = np.random.choice(choice, temporal_subset_size, replace=False)
                    else:
                        self.time_index = np.arange(0, time_range - self.cfg.seq_length)
                else:
                    time_range = next(iter(data_loaders[0]))["xd"].shape[1]
                    temporal_subset_size = self.cfg.temporal_subset[0]
                    avail_time = (time_range - self.cfg.seq_length) - temporal_subset_size
                    if avail_time > 0:
                        choice = np.arange(0, time_range - self.cfg.seq_length, 1)
                        self.time_index = np.random.choice(choice, temporal_subset_size, replace=False)
                    else:
                        self.time_index = np.arange(0, time_range - self.cfg.seq_length)
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

    def _compute_regularization(self):
        """Overloaded by the specific trainer"""
        raise NotImplementedError

    def _compute_batch_loss(
        self, prediction, target, valid_mask, target_weight, add_losses={}
    ) -> torch.Tensor:

        
        # Compute targets weighted loss. In case only one target, weight is 1
        # pred and target can be (N, C) or (N, T, C) depending on how the model is trained. 
        loss = 0
        for i, target_name in enumerate(target_weight):
            
            iypred = {}
            # If valid_mask then imask is a boolean mask (N, T). Indexing
            # the target or prediction tensors (N, T) will flatten the resulting tensor
            # to 1-D shape (N*T)[valid_mask]
            if valid_mask is not None:
                imask = valid_mask[..., i]
            else:
                imask = Ellipsis
            iytrue = target[..., i][imask] 

            if self.cfg.model_head_layer == "regression":
                iypred["y_pred"] = prediction["y_hat"][..., i][imask]
            elif self.cfg.model_head_layer == "distr_normal":
                iypred["mu"] = prediction["mu"][..., i][imask]
                iypred["sigma"] = prediction["sigma"][..., i][imask]

            w = target_weight[target_name]

            # By default it computes the average loss per sample
            loss_tmp = self.cfg.loss_fn(iytrue, **iypred)
            
            # If missing data in observation, each batch can have different number of valid samples.
            # The average loss loose the information about the size of the valid sample
            # Therefore, the loss is scaled by the fraction of valid samples in the batch
            # As the greater the size of valid samples the greater the importance in updating
            # the model parameters.
            if valid_mask is not None:
                scaling_factor = torch.sum(imask) / imask.flatten().shape[0] # fraction valid samples per batch
                loss_tmp = loss_tmp * scaling_factor

            loss = loss + loss_tmp * w

        # TODO: this is another version that should be tested! 
        # for i, target_name in enumerate(target_weight):
        #     iypred = prediction[..., i]
        #     iytrue = target[..., i]
        #     if valid_mask is not None:
        #         # here from (N, T, C) or (N, C) -> (N)
        #         imask = valid_mask[..., i]
                
        #         #iypred = iypred[imask]
        #         #iytrue = iytrue[imask]
        #         #import pdb;pdb.set_trace()
                
        #         weighting_factor = torch.sum(imask, -1) #/ n_samples # (N)
        #         iypred = torch.where(iypred.isnan(), 0, iypred) # substitute 0 where nans  
        #         iytrue = torch.where(iytrue.isnan(), 0, iytrue)

        #     # w = target_weight[target_name]

        #     # loss += w * self.cfg.loss_fn(iytrue, iypred) # (N, T)

        #     loss_tmp = self.cfg.loss_fn(iytrue, iypred) # (N, T)

        #     # mask nan 
        #     #loss_tmp[~imask] = torch.tensor([0]).float().requires_grad_()
        #     loss_tmp = loss_tmp* imask # imask, 0 non valid, 1 valid. set to zero the loss corresponsing to non valid data 

        #     # sum sequence loss 
        #     loss_sum =  torch.sum(loss_tmp, -1) # (N)

        #     if valid_mask is not None:
        #         # weighted by N.v valid samples
        #         loss_avg = (loss_sum * weighting_factor).sum() / weighting_factor.sum() # Scalar

        #         loss_avg = loss_avg / n_samples
        #     # loss to scalar and multi-variate weighting
        #     w = target_weight[target_name]
        #     #loss += w * loss_sum.mean()
        #     loss += w * loss_avg

        # compound more losses, in case dict is not empty
        #for i, (reg_func, reg_weight) in enumerate(self.add_regularization):
        #    loss += reg_func(prediction, target) * reg_weight

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

    def _concatenate_result(self, pred, target, mask=None) -> None:
        """Concatenate results for reporting and computing the metrics"""

        # prediction can be probabilistic
        if self.cfg.model_head_layer == "regression":
            pred_cpu = pred["y_hat"].detach().cpu().numpy()
        elif self.cfg.model_head_layer == "distr_normal":
            pred_cpu = pred["mu"].detach().cpu().numpy()

        target_cpu = target.detach().cpu().numpy()
        if mask is not None:
            mask_cpu = mask.detach().cpu().numpy()
        else:
            mask_cpu = mask

        if self.epoch_preds is None:
            self.epoch_preds = pred_cpu
            self.epoch_targets = target_cpu
            self.epoch_valid_masks = mask_cpu
        else:
            self.epoch_preds = np.concatenate(
                (self.epoch_preds, pred_cpu), axis=0
            )
            self.epoch_targets = np.concatenate(
                (self.epoch_targets, target_cpu), axis=0
            )
            if mask is not None:
                self.epoch_valid_masks = np.concatenate(
                    (self.epoch_valid_masks, mask_cpu), axis=0
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



    def train_valid_epoch(self, model, train_loader, val_loader, device):
        
        
        model.train()
        # set time indices for training
        # TODO: This has effect only if the trainer overload the method (i.e. for RNN)
        self._set_dynamic_temporal_downsampling([train_loader, val_loader], opt=self.optimizer)
        train_loss, train_metric = self.epoch_step(
            model, train_loader, device, opt=self.optimizer
        )

        model.eval()
        with torch.no_grad():
            # set time indices for validation
            self._set_dynamic_temporal_downsampling([train_loader, val_loader],opt=None)

            val_loss, val_metric = self.epoch_step(model, val_loader, device, opt=None)

        return train_loss, train_metric, val_loss, val_metric

    def train_epoch(self):
        pass

    def valid_epoch(self):
        pass

    def epoch_step(self):
        """Overloaded by the specific trainer"""
        pass

    def target_step(self, target, steps=1) -> torch.Tensor:
        selection = get_temporal_steps(steps)

        return target[:, selection]

    def predict_step(self, prediction, steps=-1) -> Dict[str, torch.Tensor]:
        """Return the n steps that should be predicted"""
        selection = get_temporal_steps(steps)


        output = {}
        if self.cfg.model_head_layer == "regression":
            output["y_hat"] = prediction["y_hat"][:, selection]
        elif self.cfg.model_head_layer == "distr_normal":
            for k in prediction:
                output[k] = prediction[k][:, selection]
        return output

    def save_weights(self, model, fp=None, onnx=False):
        if fp is None:
            fp = self.model_path
        if onnx:
            raise NotImplementedError()
        else:
            LOGGER.info(f"save weights to: {fp}")
            torch.save(model.state_dict(), fp)
