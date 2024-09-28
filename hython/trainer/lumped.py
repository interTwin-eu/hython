from . import *

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


