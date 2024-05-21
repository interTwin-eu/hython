import torch
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass
class BaseTrainParams:
    experiment:str
    # blabla
    loss_func:callable
    metric_func:callable
    # blabla
    train_dataloader:DataLoader
    val_dataloader:DataLoader

@dataclass
class RNNTrainParams(BaseTrainParams):
    # blabla
    subsample:bool 
    n_subsample:int
    seq_length:int
    # blabla
    time_range:int
    # blabla
    target_names:str


@dataclass
class BasinTrainParams(RNNTrainParams):
    """The loss function should be different for each model
    """
    loss_func:dict
    metric_func:dict


class BaseTrainer:

    def __init__(self, experiment):
        self.exp = experiment

    def temporal_index(self):
        pass

    def epoch_step(self):
        pass

    def predict_step(self):
        pass

    def save_weights(self, model, dp):
        torch.save(model.state_dict(), f"{dp}/{self.exp}")

def metric_epoch(metric_func, y_pred, y_true, target_names):
    metrics = metric_func(y_pred, y_true, target_names) 
    return metrics

def loss_batch(loss_func, output, target, opt=None):
    if target.shape[-1] == 1:
        target = torch.squeeze(target)
        output = torch.squeeze(output)
    
    loss = loss_func(target, output)
    if opt is not None: # evaluation
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss 

class RNNTrainer(BaseTrainer):
    
    def __init__(self, params):
        self.P = RNNTrainParams(**params)
        super(RNNTrainer, self).__init__(self.P.experiment)
        
        
    def temporal_index(self):
        """Return the temporal indices of the timeseries, it may be a subset"""
        if self.P.subsample:
            self.time_index = np.random.randint(0, self.P.time_range  - self.P.seq_length, self.P.n_subsample)
        else:
            self.time_index = np.arange(0, self.P.time_range)
        self.time_index

    def epoch_step(self, model, device, opt = None):
        if opt:
            dataloader = self.P.train_dataloader
        else:
            dataloader = self.P.val_dataloader

        running_loss = 0
        spatial_sample_size = 0 
        
        epoch_preds = None
        epoch_targets = None 
        
        for (dynamic_b, static_b, targets_b) in dataloader:

            running_time_batch_loss = 0
            for t in self.time_index: # time_index could be a subset of time indices 
                dynamic_bt = dynamic_b[:, t:(t + self.P.seq_length)].to(device)
                static_bt = static_b.to(device)
                targets_bt = targets_b[:, t:(t + self.P.seq_length)].to(device)

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

                loss_time_batch = loss_batch(self.P.loss_func, output, target, opt) 

                running_time_batch_loss += loss_time_batch

            spatial_sample_size += targets_b.size(0)

            running_loss += running_time_batch_loss

        loss = running_loss / spatial_sample_size

        metric = metric_epoch(self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names)
            
        return loss, metric
        
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

    def epoch_step(self, model, device, opt = None):
        if opt:
            dataloader = self.P.train_dataloader
        else:
            dataloader = self.P.val_dataloader

        running_loss = 0
        spatial_sample_size = 0 
        
        epoch_preds = None
        epoch_targets = None 
        
        # FORWARD DISTRIBUTED
        for (dynamic_b, static_b, targets_b) in dataloader:

            running_time_batch_loss = 0
            for t in self.time_index: # time_index could be a subset of time indices 
                dynamic_bt = dynamic_b[:, t:(t + self.P.seq_length)].to(device)
                static_bt = static_b.to(device)
                targets_bt = targets_b[:, t:(t + self.P.seq_length)].to(device)

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

                loss_time_batch = loss_batch(self.P.loss_func["distributed"], output, target, opt = None)

                running_time_batch_loss += loss_time_batch

            spatial_sample_size += targets_b.size(0)

            running_loss += running_time_batch_loss

        loss = running_loss / spatial_sample_size
        
        # FORWARD LUMPED
        y_lumped = dataloader.dataset.get_lumped_target()
        Xd_lumped = dataloader.dataset.Xd.nanmean(0, keepdim=True)
        xs_lumped = dataloader.dataset.xs.nanmean(0, keepdim=True)


        dis_running_time_batch = 0
        for t in range(self.P.time_range - self.P.seq_length):
            Xd_bt = Xd_lumped[t:(t + self.P.seq_length)]
            
            output = model(Xd_bt, xs_lumped)
             
            loss_dis_time_batch = loss_batch(self.P.loss_func["lumped"], output, y_lumped, opt=None)
            
            dis_running_time_batch += loss_dis_time_batch


        # Compound loss
        loss = 0

        if model.training:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        metric = metric_epoch(self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names)
            
        return loss, metric

class BasinDistributedTrainer(RNNTrainer):
    def __init__(self, params):
        super(BasinDistributedTrainer, self).__init__(params)

    def epoch_step(self, model, device, opt = None):
        pass