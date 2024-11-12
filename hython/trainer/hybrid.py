from . import *

class HybridTrainer(AbstractTrainer):

    def __init__(self, params: RNNTrainParams):

        self.P = params  # RNNTrainParams(**params)
        super(HybridTrainer, self).__init__(self.P.experiment)

    def temporal_index(self, data_loaders=None, opt=None):
        """Return the temporal indices of the timeseries, it may be a subset"""

        if self.P.temporal_subsampling:
            if len(self.P.temporal_subset) > 1:
                # use different time indices for training and validation

                if opt is None:
                    # validation
                    time_range = next(iter(data_loaders[-1]))[0].shape[1]
                    temporal_subset = self.P.temporal_subset[-1]
                else:
                    time_range = next(iter(data_loaders[0]))[0].shape[1]
                    temporal_subset = self.P.temporal_subset[0]

                self.time_index = np.random.randint(
                    0, time_range - self.P.seq_length, temporal_subset
                )
            else:
                # use same time indices for training and validation, time indices are from train_loader
                time_range = next(iter(data_loaders[0]))[0].shape[1]
                self.time_index = np.random.randint(
                    0, time_range - self.P.seq_length, self.P.temporal_subset[-1]
                )

        else:
            if opt is None:
                # validation
                time_range = next(iter(data_loaders[-1]))[0].shape[1]
            else:
                time_range = next(iter(data_loaders[0]))[0].shape[1]

            self.time_index = np.arange(0, time_range)

    
    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        epoch_preds = None
        epoch_targets = None
        valid_masks = None

        for forcing_b, predictor_b, target_b in dataloader: # predictors, forcings , observations

            predictor_b = predictor_b.to(device)
            flag_counter = 0                
            for t in self.time_index:
                
                target_bt = target_b[:, t : (t + self.P.seq_length)].to(device)
                
                b = self.predict_step(target_bt)
                flag = b.isnan().all().item()
                
                if flag is True: 
                    flag_counter += 1
                    #print(flag_counter)
                    continue
                
                forcing_bt = forcing_b[:, t : (t + self.P.seq_length)].to(device)

                output, param = model(predictor_b, forcing_bt)

                # print(output.mean(), param.mean(0))
                output = self.predict_step(output)
                target = self.predict_step(target_bt)

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

        # 
        epoch_loss = running_batch_loss / (len(dataloader)*len(self.time_index))
        
        metric = metric_epoch(
            self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names, valid_masks
        )
        
        return epoch_loss, metric

    def predict_step(self, arr):
        """Return the n steps that should be predicted"""
        
        return arr[:, -1] # N Ch H W  