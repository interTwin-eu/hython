from . import *


class ConvTrainer(AbstractTrainer):
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
