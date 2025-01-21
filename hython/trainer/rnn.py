from . import *


class RNNTrainer(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(RNNTrainer, self).__init__()
        self.cfg = cfg

    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        data_points = 0

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
                
                batch_sequence_loss = self._compute_batch_loss(
                    prediction=output,
                    target=target,
                    valid_mask=None,
                    target_weight=self.target_weights,
                )

                self._backprop_loss(batch_sequence_loss, opt)

                batch_temporal_loss += batch_sequence_loss.detach()

            data_points += data["xd"].size(0)

            running_batch_loss += batch_temporal_loss

        epoch_loss = running_batch_loss / data_points

        metric = self._compute_metric()

        return epoch_loss, metric
