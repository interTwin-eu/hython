from . import *


class CalTrainer(AbstractTrainer):
    def __init__(self, cfg):
        super(CalTrainer, self).__init__()
        self.cfg = cfg
        self.cfg["target_weight"] = {t: 1 / len(self.cfg.target_variables) for t in self.cfg.target_variables}
        
    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        epoch_preds = None
        epoch_targets = None
        valid_masks = None

        for data in dataloader:  # predictors, forcings , observations
            predictor_b = data["xs"].to(device)
            target_b = data["y"].to(device)
            forcing_b = data["xd"].to(device)

            pred = model(predictor_b, forcing_b)

            output = self.predict_step(pred["y_hat"])
            target = self.predict_step(target_b)

            valid_mask = ~target.isnan()  # non null values

            scaling_factor = sum(valid_mask) / len(
                target
            )  # scale by number of valid samples in a mini-batch

            mini_batch_loss = loss_batch(
                    loss_func=self.cfg.loss_fn,
                    output= output,
                    target=target,
                    opt=opt,
                    gradient_clip= self.cfg.gradient_clip,
                    model=model,
                    valid_mask=valid_mask,
                    #add_losses=add_losses,
                    target_weight=self.cfg.target_weight,
                    scaling_factor=scaling_factor,
            )

            if epoch_preds is None:
                epoch_preds = output.detach().cpu().numpy()
                epoch_targets = target.detach().cpu().numpy()
                valid_masks = valid_mask.detach().cpu().numpy()
            else:
                epoch_preds = np.concatenate(
                    (epoch_preds, output.detach().cpu().numpy()), axis=0
                )
                epoch_targets = np.concatenate(
                    (epoch_targets, target.detach().cpu().numpy()), axis=0
                )
                valid_masks = np.concatenate(
                    (valid_masks, valid_mask.detach().cpu().numpy()), axis=0
                )

            # Accumulate mini-batch loss, only valid samples
            running_batch_loss += mini_batch_loss

        epoch_loss = running_batch_loss / len(dataloader)

        metric = metric_epoch(
            self.cfg.metric_fn,
            epoch_targets,
            epoch_preds,
            self.cfg.target_variables,
            valid_masks,
        )

        return epoch_loss, metric

    def predict_step(self, arr):
        """Return the n steps that should be predicted"""

        return arr[:, -1]  # N Ch H W
