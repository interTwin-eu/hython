from . import *


class RNNTrainer(AbstractTrainer):
    def __init__(self, params: RNNTrainParams):
        """Train architecture that

        Parameters
        ----------
        params : RNNTrainParams

        """
        self.P = params  # RNNTrainParams(**params)
        super(RNNTrainer, self).__init__(self.P.experiment)

    def temporal_index(self, data_loaders=None, opt=None):
        """Return the temporal indices of the timeseries, it may be a subset"""

        if self.P.temporal_subsampling:
            if len(self.P.temporal_subset) > 1:
                # use different time indices for training and validation

                if opt is None:
                    # validation
                    time_range = next(iter(data_loaders[-1]))["xd"].shape[1]
                    temporal_subset = self.P.temporal_subset[-1]
                else:
                    time_range = next(iter(data_loaders[0]))["xd"].shape[1]
                    temporal_subset = self.P.temporal_subset[0]

                self.time_index = np.random.randint(
                    0, time_range - self.P.seq_length, temporal_subset
                )
            else:
                # use same time indices for training and validation, time indices are from train_loader
                time_range = next(iter(data_loaders[0]))["xd"].shape[1]
                self.time_index = np.random.randint(
                    0, time_range - self.P.seq_length, self.P.temporal_subset[-1]
                )

        else:
            if opt is None:
                # validation
                time_range = next(iter(data_loaders[-1]))["xd"].shape[1]
            else:
                time_range = next(iter(data_loaders[0]))["xd"].shape[1]

            self.time_index = np.arange(0, time_range)

    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        data_points = 0

        epoch_preds = None
        epoch_targets = None

        for data in dataloader:
            batch_temporal_loss = 0

            for t in self.time_index:  # time_index could be a subset of time indices
                # filter sequence
                dynamic_bt = data["xd"][:, t : (t + self.P.seq_length)].to(device)
                targets_bt = data["y"][:, t : (t + self.P.seq_length)].to(device)

                # static --> dynamic size (repeat time dim)
                static_bt = (
                    data["xs"].unsqueeze(1).repeat(1, dynamic_bt.size(1), 1).to(device)
                )

                x_concat = torch.cat(
                    (dynamic_bt, static_bt),
                    dim=-1,
                )

                pred = model(x_concat)

                lstm_output = pred["y_hat"]
                                   
                # physics based loss
                add_losses = self.P.loss_physics_collection["PrecipSoilMoisture"](
                    targets_bt[..., [0]], lstm_output[..., [0]]
                )

                output = self.predict_step(lstm_output, steps=-1)
                target = self.predict_step(targets_bt, steps=-1)

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
                    self.P.loss_func,
                    output,
                    target,
                    opt,
                    self.P.gradient_clip,
                    model,
                    valid_mask = None, 
                    add_losses = add_losses,
                    target_weight=self.P.target_weight
                )

                batch_temporal_loss += batch_sequence_loss

            data_points += data["xd"].size(0)

            running_batch_loss += batch_temporal_loss

        epoch_loss = running_batch_loss / data_points

        metric = metric_epoch(
            self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names
        )

        return epoch_loss, metric

    def predict_step(self, arr, steps=-1):
        """Return the n steps that should be predicted"""
        return arr[:, steps]
