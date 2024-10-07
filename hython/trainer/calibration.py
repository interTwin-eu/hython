from . import *
   
class ParameterLearningTrainer(AbstractTrainer):

    def __init__(self, params: RNNTrainParams):

        self.P = params  # RNNTrainParams(**params)
        super(ParameterLearningTrainer, self).__init__(self.P.experiment)


    def epoch_step(self, model, dataloader, device, opt=None):
        running_batch_loss = 0
        data_points = 0

        epoch_preds = None
        epoch_targets = None
        valid_masks = None

        # N T C H W
        flag_counter = 0
        for predictor_b, forcing_b, target_b in dataloader: # predictors, forcings , observations
            
            #TODO: this is a temporary solution, to be removed!!
            cube_slice = self.predict_step(target_b) # N C H W
            flag = cube_slice.isnan().all().item()
            if flag is True: 
                flag_counter += 1
                print(flag_counter)
                continue

            target_b = target_b.to(device)

            forcing_b = forcing_b.to(device)

            predictor_b = predictor_b.to(device)
            
            # Predict wflow parameters
            parameter = model["transfer_nn"](predictor_b).to(device) #  N C H W -> N C h w   #   N L H W Cout

            
            # minmax scaling as surrogate expects minmax scaled effective parameters
            # TODO: should I compute the minmax over the mini-batch? should I compute the minmax by using global statistics?
            #parameter = (parameter - torch.amin(parameter, dim=(0,2,3), keepdim=True) ) / (torch.amax(parameter, dim=(0,2,3), keepdim=True) - torch.amin(parameter, dim=(0,2,3) , keepdim=True))

            parameter = (self.P.stats[1]*parameter) + self.P.stats[0]
            # concat dynamic and static parameters, common to convLSTM and LSTM
            # if both dynamic?
            # convLSTM: N C H W -> N L C H W 
            # LSTM:  NCHW -> NC -> NLC
            #
            X = torch.concat([
                            parameter.unsqueeze(1).repeat(1, forcing_b.size(1), 1,1,1),
                            forcing_b], dim=2)

            output = model["surrogate"](X)[0] # convLSTM: NLCHW, LSTM: NLC
            output = torch.permute(output, (0, 1, 4, 2, 3)) # N L C H W 
            #import pdb;pdb.set_trace()
            # flatten HW
            output = self.predict_step(output).flatten(2)
            target = self.predict_step(target_b).flatten(2)

            valid_mask = ~target.isnan() # non null values

            batch_sequence_loss = loss_batch(self.P.loss_func, output, target, opt, self.P.gradient_clip, model, valid_mask)
            
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

            

            running_batch_loss += batch_sequence_loss.detach()

        epoch_loss = running_batch_loss / (len(dataloader) - flag_counter)
        metric = metric_epoch(
            self.P.metric_func, epoch_targets, epoch_preds, self.P.target_names, valid_masks
        )

        return epoch_loss, metric

    def predict_step(self, arr):
        """Return the n steps that should be predicted"""
        
        return arr[:, -1] # N Ch H W  