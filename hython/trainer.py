import torch
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from torch.nn.modules.loss import _Loss
from hython.hython.metrics import Metric

import copy
import torch
import numpy as np
from tqdm.auto import tqdm


class BaseTrainParams:
    pass


class RNNTrainParams(BaseTrainParams):
    def __init__(
        self,
        loss_func: _Loss,
        metric_func: Metric,
        experiment: str,
        temporal_subsampling: bool,
        temporal_subset: int,
        seq_length: int,
        target_names: list,
    ):
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.experiment = experiment
        self.temporal_subsampling = temporal_subsampling
        self.temporal_subset = temporal_subset
        self.seq_length = seq_length
        self.target_names = target_names


@dataclass
class BasinTrainParams(RNNTrainParams):
    """The loss function should be different for each model"""

    loss_func: _Loss
    metric_func: Metric


class AbstractTrainer(ABC):
    def __init__(self, experiment: str):
        self.exp = experiment

    def temporal_index(self, time_range):
        pass

    def epoch_step(self):
        pass

    def predict_step(self):
        pass

    def save_weights(self, model, fp, onnx=False):
        if onnx:
            raise NotImplementedError()
        else:
            torch.save(model.state_dict(), fp)


def metric_epoch(metric_func, y_pred, y_true, target_names):
    metrics = metric_func(y_pred, y_true, target_names)
    return metrics


def loss_batch(loss_func, output, target, opt=None):
    if target.shape[-1] == 1:
        target = torch.squeeze(target)
        output = torch.squeeze(output)

    loss = loss_func(target, output)
    if opt is not None:  # evaluation
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss


class RNNTrainer(AbstractTrainer):
    def __init__(self, params: RNNTrainParams):
        self.P = params  # RNNTrainParams(**params)
        print(self.P)
        super(RNNTrainer, self).__init__(self.P.experiment)

    def temporal_index(self, time_range=None):
        """Return the temporal indices of the timeseries, it may be a subset"""

        if time_range is not None:
            if self.P.temporal_subsampling:
                self.time_index = np.random.randint(
                    0, time_range - self.P.seq_length, self.P.temporal_subset
                )
            else:
                self.time_index = np.arange(0, time_range)
        else:
            self.time_index = None

    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        data_points = 0

        epoch_preds = None
        epoch_targets = None

        for dynamic_b, static_b, targets_b in dataloader:
            batch_temporal_loss = 0

            for t in self.time_index:  # time_index could be a subset of time indices
                dynamic_bt = dynamic_b[:, t : (t + self.P.seq_length)].to(device)
                static_bt = static_b.to(device)
                targets_bt = targets_b[:, t : (t + self.P.seq_length)].to(device)

                output = model(dynamic_bt, static_bt)

                output = self.predict_step(output)
                target = self.predict_step(targets_bt)

                if epoch_preds is None:
                    epoch_preds = output.detach().cpu().numpy()
                    epoch_targets = target.detach().cpu().numpy()
                else:
                    epoch_preds = np.concatenate(
                        (epoch_preds, output.detach().cpu().numpy()), axis=0
                    )
                    epoch_targets = np.concatenate(
                        (epoch_targets, target.detach().cpu().numpy()), axis=0
                    )

                batch_sequence_loss = loss_batch(self.P.loss_func, output, target, opt)

                batch_temporal_loss += batch_sequence_loss

            data_points += targets_b.size(0)

            running_batch_loss += batch_temporal_loss

        epoch_loss = running_batch_loss / data_points

        metric = metric_epoch(
            self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names
        )

        return epoch_loss, metric

    def predict_step(self, arr):
        """Return the n steps that should be predicted"""
        return arr[:, -1]


class BasinLumpedTrainer(RNNTrainer):
    def __init__(self, params):
        super(BasinLumpedTrainer, self).__init__(params)

    def forward_distributed(self):
        pass

    def forward_lumped(self):
        pass

    def epoch_step(self, model, device, opt=None):
        if opt:
            dataloader = self.P.train_dataloader
        else:
            dataloader = self.P.val_dataloader

        running_batch_loss = 0
        data_points = 0

        epoch_preds = None
        epoch_targets = None

        # FORWARD DISTRIBUTED
        for dynamic_b, static_b, targets_b in dataloader:
            batch_temporal_loss = 0

            for t in self.time_index:  # time_index could be a subset of time indices
                dynamic_bt = dynamic_b[:, t : (t + self.P.seq_length)].to(device)
                static_bt = static_b.to(device)
                targets_bt = targets_b[:, t : (t + self.P.seq_length)].to(device)

                output = model(dynamic_bt, static_bt)

                output = self.predict_step(output)
                target = self.predict_step(targets_bt)

                if epoch_preds is None:
                    epoch_preds = output.detach().cpu().numpy()
                    epoch_targets = target.detach().cpu().numpy()
                else:
                    epoch_preds = np.concatenate(
                        (epoch_preds, output.detach().cpu().numpy()), axis=0
                    )
                    epoch_targets = np.concatenate(
                        (epoch_targets, target.detach().cpu().numpy()), axis=0
                    )

                batch_sequence_loss = loss_batch(
                    self.P.loss_func["distributed"], output, target, opt=None
                )

                batch_temporal_loss += batch_sequence_loss

            data_points += targets_b.size(0)

            running_batch_loss += batch_temporal_loss

        epoch_loss = running_batch_loss / data_points

        # FORWARD LUMPED
        y_lumped = dataloader.dataset.get_lumped_target()
        Xd_lumped = dataloader.dataset.Xd.nanmean(0, keepdim=True)
        xs_lumped = dataloader.dataset.xs.nanmean(0, keepdim=True)

        dis_running_time_batch = 0
        for t in range(self.P.time_range - self.P.seq_length):
            Xd_bt = Xd_lumped[t : (t + self.P.seq_length)]

            output = model(Xd_bt, xs_lumped)

            loss_dis_time_batch = loss_batch(
                self.P.loss_func["lumped"], output, y_lumped, opt=None
            )

            dis_running_time_batch += loss_dis_time_batch

        # Compound loss
        loss = 0

        if model.training:
            opt.zero_grad()
            loss.backward()
            opt.step()

        metric = metric_epoch(
            self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names
        )

        return loss, metric


class BasinDistributedTrainer(RNNTrainer):
    def __init__(self, params):
        super(BasinDistributedTrainer, self).__init__(params)

    def epoch_step(self, model, device, opt=None):
        pass


class BasinGraphTrainer(RNNTrainer):
    def __init__(self, params):
        super(BasinGraphTrainer, self).__init__(params)

    def epoch_step(self, model, device, opt=None):
        pass


def train_val(
    trainer,
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    lr_scheduler,
    dp_weights,
    device,
    time_range=None,
):
    target_names = trainer.P.target_names

    loss_history = {"train": [], "val": []}
    metric_history = {f"train_{target}": [] for target in target_names}
    metric_history.update({f"val_{target}": [] for target in target_names})

    best_loss = float("inf")

    for epoch in tqdm(range(epochs)):
        trainer.temporal_index(time_range)  # For RNNs based models

        model.train()

        train_loss, train_metric = trainer.epoch_step(
            model, train_loader, device, opt=optimizer
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = trainer.epoch_step(
                model, val_loader, device, opt=None
            )

        lr_scheduler.step(val_loss)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)

        for target in target_names:
            metric_history[f"train_{target}"].append(train_metric[target])
            metric_history[f"val_{target}"].append(val_metric[target])

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            trainer.save_weights(model, dp_weights)
            print("Copied best model weights!")

        print(f"train loss: {train_loss}")
        print(f"val loss: {val_loss}")

    model.load_state_dict(best_model_weights)

    return model, loss_history, metric_history
