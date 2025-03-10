from . import *


class CalTrainer(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(CalTrainer, self).__init__(cfg=cfg)

    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0

        for data in dataloader:
            predictor_b = data["xs"].to(device)
            target_b = data["y"].to(device)
            forcing_b = data["xd"].to(device)

            pred = model(predictor_b, forcing_b)
            
            output = self.predict_step(pred, steps=self.cfg.predict_steps)
            target = self.target_step(target_b, steps=self.cfg.predict_steps)
            # TODO: consider moving missing values loss handling in the compute loss method
            valid_mask = ~target.isnan()  # non null values

            self._concatenate_result(output, target, valid_mask)

            # Compute loss: default returns average loss per sample
            mini_batch_loss = self._compute_batch_loss(
                prediction=output,
                target=target,
                valid_mask=valid_mask,
                target_weight=self.target_weights,
            )
            if self.cfg.predict_steps != 0:
                mini_batch_loss = mini_batch_loss.mean()
            self._backprop_loss(mini_batch_loss, opt)

            # Accumulate mini-batch loss, only valid samples
            running_batch_loss += mini_batch_loss.detach()

        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric
