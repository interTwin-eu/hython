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

        # model file

        if self.cfg.model_file_name is not None:
            self.model_path = f"{self.run_dir}/{self.cfg.model_file_name}"
        else:
            self.model_path = f"{self.run_dir}/model.pt"

        LOGGER.info(f"Run directory: {self.run_dir}") 
        LOGGER.info(f"Model path: {self.model_path}") 
        

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
    ) -> torch.Tensor:

        
        # Compute targets weighted loss. In case only one target, weight is 1
        # pred and target can be (N, C) or (N, T, C) depending on how the model is trained. 
        loss = 0
        for i, target_name in enumerate(target_weight):
            
            iypred = {}

            if valid_mask is not None:
                imask = valid_mask[..., i]
            else:
                imask = Ellipsis
            
            # target
            iytrue = target[..., i][imask] 

            if self.cfg.model_head_layer == "regression":
                iypred["y_pred"] = prediction["y_hat"][..., i][imask]
                n = torch.ones_like(iypred["y_pred"])
            elif self.cfg.model_head_layer == "distr_normal":
                iypred["mu"] = prediction["mu"][..., i][imask]
                iypred["sigma"] = prediction["sigma"][..., i][imask]
                n = torch.ones_like(iypred["mu"])

            #iypred = pred[..., i]
            #iytrue = target[..., i]
            # if valid_mask is not None:
            #     n = torch.ones_like(iypred)
            #     imask = valid_mask[..., i]
            #     iypred = iypred[imask]
            #     iytrue = iytrue[imask]

            w = target_weight[target_name]
            # if isinstance(self.model.head, RegressionHead):
            #     loss_tmp = self.cfg.loss_fn(iytrue, iypred)
            # else:
            loss_tmp = self.cfg.loss_fn(iytrue, **iypred)

            # in case there are missing observations in the batch
            # the loss should be weighted to reduce the importance
            # of the loss on the update of the NN weights
            if valid_mask is not None:
                scaling_factor = torch.sum(imask) / torch.sum(n) # fraction valid samples per batch
                # scale by number of valid samples in a mini-batch
                loss_tmp =  loss_tmp * scaling_factor
            
            loss =+ loss_tmp * w

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


        self._set_regularization()

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
