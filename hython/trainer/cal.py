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
        self.cfg["target_weight"] = {t: 1 / len(self.cfg.target_variables) for t in self.cfg.target_variables}
        
    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0


        for data in dataloader:  # predictors, forcings , observations
            predictor_b = data["xs"].to(device)
            target_b = data["y"].to(device)
            forcing_b = data["xd"].to(device)

            pred = model(predictor_b, forcing_b)

            output = self.predict_step(pred["y_hat"], steps=-1)
            target = self.predict_step(target_b, steps=-1)

            valid_mask = ~target.isnan()  # non null values

            scaling_factor = sum(valid_mask) / len(
                target
            )  # scale by number of valid samples in a mini-batch

            self._concatenate_result(output, target, valid_mask)

            mini_batch_loss = self._compute_loss(
                    output= output,
                    target=target,
                    valid_mask=valid_mask,
                    target_weight=self.cfg.target_weight,
                    scaling_factor=scaling_factor,
            )

            self._backprop_loss(mini_batch_loss, opt)

            # Accumulate mini-batch loss, only valid samples
            running_batch_loss += mini_batch_loss

        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric