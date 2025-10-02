from . import *


class RNNTrainer(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(RNNTrainer, self).__init__(cfg=cfg)

    def _compute_regularization(self, target):
        if self.cfg.regularization is not None:
            return self.cfg.regularization(target)
        else:
            return 0
        
    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0

        for data in dataloader:
            batch_temporal_loss = 0

            for t in self.time_index:  # time_index could be a subset of time indices
                # filter sequence
                dynamic_bt = data["xd"][:, t : (t + self.cfg.seq_length)].to(device)
                targets_bt = data["y"][:, t : (t + self.cfg.seq_length)].to(device)

                # static --> dynamic size (repeat time dim)
                static_bt = (
                    data["xs"].unsqueeze(1).repeat(1, dynamic_bt.size(1), 1).to(device)
                )

                x_concat = torch.cat(
                    (dynamic_bt, static_bt),
                    dim=-1,
                )

                pred = model(x_concat)

                output = self.predict_step(pred, steps=self.cfg.predict_steps)
                target = self.target_step(targets_bt, steps=self.cfg.predict_steps)
                
                self._concatenate_result(output, target) 

                # Compute loss: default returns average loss per sample
                batch_sequence_loss = self._compute_batch_loss(
                    prediction=output,
                    target=target,
                    valid_mask=None,
                    target_weight=self.target_weights,
                )


                # Add regularization 
                reg_loss = self._compute_regularization(target)

                batch_sequence_loss = batch_sequence_loss + reg_loss

                self._backprop_loss(batch_sequence_loss, opt)
                
                batch_temporal_loss += batch_sequence_loss.detach()
            
            batch_temporal_loss /= len(self.time_index)

            running_batch_loss += batch_temporal_loss
        
        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric


class RNNTrainerHPC(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(RNNTrainerHPC, self).__init__(cfg=cfg)

    def _compute_regularization(self, target):
        if self.cfg.regularization is not None:
            return self.cfg.regularization(target)
        else:
            return 0
        
    def epoch_step(self, model, dataloader, device, opt=None):
        running_loss = 0

        for data in dataloader:            
            static_bt = (
                data["xs"].unsqueeze(1).repeat(1, data["xd"].size(1), 1).to(device)
            )

            x_concat = torch.cat(
                (data["xd"].to(device), static_bt),
                dim=-1,
            )

            pred = model(x_concat)

            output = self.predict_step(pred, steps=self.cfg.predict_steps)
            target = self.target_step(data["y"].to(device), steps=self.cfg.predict_steps)

            self._concatenate_result(output, target) 

            # Compute loss: default returns average loss per sample
            batch_loss = self._compute_batch_loss(
                prediction=output,
                target=target,
                valid_mask=None,
                target_weight=self.target_weights,
            )

            # Add regularization
            reg_loss = self._compute_regularization(target)

            batch_loss = batch_loss + reg_loss

            self._backprop_loss(batch_loss, opt)
            
            running_loss += batch_loss.detach()
            
        epoch_loss = running_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric


    def _set_dynamic_temporal_downsampling(self, data_loaders=None, opt=None):
        """Return the temporal indices of the timeseries, it may be a subset"""

        try:
            temporal_downsampling = self.cfg.temporal_downsampling
        except:
            return 
            #temporal_downsampling = False
            
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
