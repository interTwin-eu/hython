from . import *

class CalTrainer(AbstractTrainer):

    def __init__(self, params: RNNTrainParams):

        self.P = params  # RNNTrainParams(**params)
        super(CalTrainer, self).__init__(self.P.experiment)
    
    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        epoch_preds = None
        epoch_targets = None
        valid_masks = None

        for forcing_b, predictor_b, target_b in dataloader: # predictors, forcings , observations

            predictor_b = predictor_b.to(device)
            target_b = target_b.to(device)
            forcing_b = forcing_b.to(device)

            output, param = model(predictor_b, forcing_b)

            # print(output.mean(), param.mean(0))
            output = self.predict_step(output)
            target = self.predict_step(target_b)

            valid_mask = ~target.isnan() # non null values

            scaling_factor = sum(valid_mask)/len(target) # scale by number of valid samples in a mini-batch
            
            mini_batch_loss = loss_batch(self.P.loss_func, output, target, opt, self.P.gradient_clip, model, valid_mask, scaling_factor = scaling_factor)

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
                valid_masks = np.concatenate( (
                        valid_masks, valid_mask.detach().cpu().numpy()), axis=0)
            
            # Accumulate mini-batch loss, only valid samples   
            running_batch_loss += mini_batch_loss

        epoch_loss = running_batch_loss / len(dataloader)
        
        metric = metric_epoch(
            self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names, valid_masks
        )
        
        return epoch_loss, metric

    def predict_step(self, arr):
        """Return the n steps that should be predicted"""
        
        return arr[:, -1] # N Ch H W  