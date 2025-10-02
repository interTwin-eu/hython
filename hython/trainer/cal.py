from . import *


class CalTrainer(AbstractTrainer):
    """

    Parameters
    ----------
    cfg:

    """

    def __init__(self, cfg):
        super(CalTrainer, self).__init__(cfg=cfg)

    def _compute_regularization(self, param, regularizer):
        if self.cfg.regularization is not None:
            return self.cfg.regularization[0][regularizer](param)
        else:
            return 0
    
    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        # The model may predict one or many targets (N). 
        # The calibration may be performed on one to N targets.
        # Needs to subset the model output to match calibration 
        index_tensor_pred = [self.cfg.head_output_variables.index(t) for t in self.cfg.target_variables]
        for data in dataloader:
            predictor_b = data["xs"].to(device)
            target_b = data["y"].to(device)
            forcing_b = data["xd"].to(device)

            head_b = data["xp"].to(device)

            pred = model(predictor_b, forcing_b, head_b) # surrogate prediction

            output = self.predict_step(pred, steps=self.cfg.predict_steps, subset_index=index_tensor_pred)
            target = self.target_step(target_b, steps=self.cfg.predict_steps)
            # 
            
            #import pdb; pdb.set_trace()
            # rescale surrogate output to same simulation as target was rescaled
            #output["y_hat"] = dataloader.dataset.sim_std.to(device)/output["y_hat"].std() * (output["y_hat"] - output["y_hat"].mean()) + dataloader.dataset.sim_mean.to(device)

            # TODO: consider moving missing values loss handling in the compute loss method
            valid_mask = ~target.isnan()  # non null values
           
            self._concatenate_result(output, target, valid_mask, param = pred["param"])
            
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

            # Add regularization acting on parameters
            reg_loss = self._compute_regularization(pred["param"], "param_bound_check")
            # Add regularization on outputs
            reg_loss2 = self._compute_regularization(pred["y_hat"], "target_bound_check")            
            #print(reg_loss, reg_loss2)
            loss = mini_batch_loss + reg_loss #+ reg_loss2

            self._backprop_loss(loss, opt)

            # Accumulate mini-batch loss, only valid samples
            running_batch_loss += loss.detach()
            #print(loss)
        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        self._log_calib_parameters(opt)

        return epoch_loss, metric
