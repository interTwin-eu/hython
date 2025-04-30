from . import *


class RNNTrainer(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(RNNTrainer, self).__init__(cfg=cfg)

    def epoch_step(self, model, dataloader, device, opt=None):
        running_loss = 0

        for data in dataloader:            
            #for t in self.time_index:  # time_index could be a subset of time indices
                # filter sequence
            #    dynamic_bt = data["xd"][:, t : (t + self.cfg.seq_length)].to(device)
            #    targets_bt = data["y"][:, t : (t + self.cfg.seq_length)].to(device)

                # static --> dynamic size (repeat time dim)
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

            self._backprop_loss(batch_loss, opt)
            
            running_loss += batch_loss.detach()
            
        epoch_loss = running_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric
