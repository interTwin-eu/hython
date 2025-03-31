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

            running_batch_loss += batch_loss.detach()

        epoch_loss = running_batch_loss / len(dataloader)

        metric = self._compute_metric()

        return epoch_loss, metric
