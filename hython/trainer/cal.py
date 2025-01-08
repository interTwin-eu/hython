from . import *


class CalTrainer(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(CalTrainer, self).__init__()
        self.cfg = cfg
        # self.cfg["target_weight"] = {t: 1 / len(self.cfg.target_variables) for t in self.cfg.target_variables}

    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0

        for data in dataloader:
            predictor_b = data["xs"].to(device)
            target_b = data["y"].to(device)
            forcing_b = data["xd"].to(device)

            pred = model(predictor_b, forcing_b)

            output = self.predict_step(pred["y_hat"], steps=-1)
            target = self.predict_step(target_b, steps=-1)

            # TODO: consider moving missing values loss handling in the compute loss method
            valid_mask = ~target.isnan()  # non null values

            self._concatenate_result(output, target, valid_mask)

            mini_batch_loss = self._compute_batch_loss(
                prediction=output,
                target=target,
                valid_mask=valid_mask,
                target_weight=self.target_weights,
            )

            #import pdb;pdb.set_trace()
            #mini_batch_loss = mini_batch_loss.mean()
            self._backprop_loss(mini_batch_loss, opt)

            # Accumulate mini-batch loss, only valid samples
            running_batch_loss += mini_batch_loss

        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric
