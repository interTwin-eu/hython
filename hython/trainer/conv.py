from . import *

class ConvTrainer(AbstractTrainer):
    def __init__(self, cfg):
        super(ConvTrainer, self).__init__(cfg = cfg)

    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0

        # N T C H W
        for data in dataloader:
            static_b = data["xs"].unsqueeze(1).repeat(1, data["xd"].size(1), 1, 1, 1).to(device)
            
            x_concat = torch.concat([data["xd"].to(device), static_b], 2).to(device)

            pred = model(x_concat)  # # N L H W Cout, # 

            output = self.predict_step(pred, steps=self.cfg.predict_steps) # N L H W Cout  => # N H W C

             
            if self.cfg.model_head_layer == "regression":
                output["y_hat"] = torch.permute(output["y_hat"], (0, 3, 1, 2)).flatten(2)  # N H W C => N C H W => N C Pixel
            else:
                output["mu"] = torch.permute(output["mu"], (0, 3, 1, 2)).flatten(2)
                output["sigma"] = torch.permute(output["sigma"], (0, 3, 1, 2)).flatten(2)
            
            target = self.target_step(data["y"].to(device), steps=self.cfg.predict_steps).flatten(2) # N L H W Cout  => # N H W C => N C Pixel

            self._concatenate_result(output, target) 

            target = torch.permute(target, (0,2,1)).flatten(0,1) # N C Pixel > N Pixel C > Pixel C

            if self.cfg.model_head_layer == "regression":
                output["y_hat"] = torch.permute(output["y_hat"], (0,2,1)).flatten(0,1)  # N C Pixel > N Pixel C > Pixel C
            else:
                output["mu"] = torch.permute(output["mu"], (0,2,1)).flatten(0,1) # N C Pixel > N Pixel C > Pixel C
                output["sigma"] = torch.permute(output["sigma"], (0,2,1)).flatten(0,1)

            batch_loss = self._compute_batch_loss(
                prediction=output,
                target=target,
                valid_mask=None,
                target_weight=self.target_weights,
            )

            self._backprop_loss(batch_loss, opt)

            running_batch_loss += batch_loss

        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric



class ConvTrainer2(AbstractTrainer):
    def __init__(self, params):
        self.P = params  # RNNTrainParams(**params)
        super(ConvTrainer, self).__init__(self.P.experiment)

    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        data_points = 0

        epoch_preds = None
        epoch_targets = None

        # N T C H W
        for dynamic_b, static_b, targets_b in dataloader:
            targets_b = targets_b.to(device)
            if len(static_b[0]) > 1:
                input = torch.concat([dynamic_b, static_b], 2).to(device)
            else:
                input = dynamic_b.to(device)
            #
            output = model(input)[0]  # # N L H W Cout
            output = torch.permute(output, (0, 1, 4, 2, 3))  # N L C H W

            # physics loss
            add_losses = self.P.loss_physics_collection["PrecipSoilMoisture"](
                input[..., [0]], output[..., [0]]
            )

            output = self.predict_step(output).flatten(
                2
            )  # N L C H W  => # N C H W => N C Pixel
            target = self.predict_step(targets_b).flatten(2)

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
                valid_mask=None,
                add_losses=add_losses,
            )

            running_batch_loss += batch_sequence_loss.detach()

        epoch_loss = running_batch_loss / len(dataloader)

        metric = metric_epoch(
            self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names
        )

        return epoch_loss, metric

    def predict_step(self, arr):
        """Return the n steps that should be predicted"""

        return arr[:, -1]  # N Ch H W
