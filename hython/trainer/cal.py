from . import *


class CalTrainer(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(CalTrainer, self).__init__(cfg=cfg)

    def _compute_regularization(self, param):
        if self.cfg.regularization is not None:
            return self.cfg.regularization(param)
        else:
            return 0
    
    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        # The model may predict one or many targets (N). 
        # The calibration may be performed on one to N targets.
        # Needs to subset the model output to match calibration 
        index_tensor_pred = [self.cfg.head_target_variables.index(t) for t in self.cfg.calibration_target_variables]
        for data in dataloader:
            predictor_b = data["xs"].to(device)
            target_b = data["y"].to(device)
            forcing_b = data["xd"].to(device)

            pred = model(predictor_b, forcing_b) # surrogate prediction
            
            output = self.predict_step(pred, steps=self.cfg.predict_steps, subset_index=index_tensor_pred)
            target = self.target_step(target_b, steps=self.cfg.predict_steps)

            # subset model output
            
            
            # TODO: consider moving missing values loss handling in the compute loss method
            valid_mask = ~target.isnan()  # non null values
            
            self._concatenate_result(output, target, valid_mask)
            
            # Compute loss: default returns average loss per sample
            mini_batch_loss = self._compute_batch_loss(
                prediction=output,
                target=target,
                valid_mask=valid_mask,
                target_weight=self.target_weights,
                #calibration_vars=self.cfg.target_variables, # In case
            )
            
            if self.cfg.predict_steps != 0: # not necessary as the loss is already averaged
                mini_batch_loss = mini_batch_loss.mean()

            # Add regularization 
            reg_loss = self._compute_regularization(pred["param"])
            loss = mini_batch_loss + reg_loss

            self._backprop_loss(loss, opt)

            # Accumulate mini-batch loss, only valid samples
            running_batch_loss += mini_batch_loss.detach()

        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric
