import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# get the current learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


# Define the Loss_batch function
def loss_batch(loss_func, output, target, opt=None):
    if target.shape[-1] == 1:
        target = torch.squeeze(target)
        output = torch.squeeze(output)
    
    loss = loss_func(target, output)
    if opt is not None: # evaluation
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss 


# Define the metric_epoch function
def metric_epoch(metric_func, y_pred, y_true, target_names):
    metrics = metric_func(y_pred, y_true, target_names) 
    return metrics


# Define the loss_epoch function
def loss_epoch(model, loss_func, metric_func, dataset_dl, target_names, device, opt=None, ts_idx= None, seq_length = None):
    running_loss = 0
    
    spatial_sample_size = 0 
    
    epoch_preds = None
    epoch_targets = None 

    for (
        predictors_b,
        static_params_b,
        targets_b,
    ) in dataset_dl:  # batch_predictors, batch_static_params, batch_targets
        
        running_time_batch_loss = 0

        for t in ts_idx:

            predictors_bt = predictors_b[:, t:(t + seq_length)].to(device)
            static_params_bt = static_params_b.to(device)
            targets_bt = targets_b[:, t:(t + seq_length)].to(device)
            
            output = model(predictors_bt, static_params_bt)[:, -1] # take last time step

            if epoch_preds is None:
                epoch_preds = output.detach().cpu().numpy()
                epoch_targets = targets_bt[:, -1].detach().cpu().numpy()
            else:
                epoch_preds = np.concatenate(
                    (epoch_preds, output.detach().cpu().numpy()), axis=0
                )
                epoch_targets = np.concatenate(
                    (epoch_targets, targets_bt[:, -1].detach().cpu().numpy()), axis=0
                )

            # get loss per i time batch
            loss_time_batch = loss_batch(loss_func, output, targets_bt[:, -1], opt) 

            # update running loss
            running_time_batch_loss += loss_time_batch #* targets_bt.size(0)

        #running_time_batch_loss /= len(ts_idx)
        
        # accumulate number of samples
        spatial_sample_size += targets_b.size(0)

        running_loss += running_time_batch_loss

    # average loss value
    loss = running_loss / spatial_sample_size

    # average metric value
    metric = metric_epoch(metric_func, epoch_targets, epoch_preds, target_names)
    
    return loss, metric


def train_val(model, params):
    num_epochs = params["num_epochs"]
    seq_length = params["seq_length"]
    temporal_sampling_size = params["temporal_sampling_size"]
    temporal_sampling_epoch = params["temporal_sampling_idx_change_with_epoch"]
    ts_range = params["ts_range"]
    loss_func = params["loss_func"]
    metric_func = params["metric_func"]
    target_names = params["target_names"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    device = params["device"]

    loss_history = {"train": [], "val": []}

    metric_history = {f'train_{t}': [] for t in target_names}
    metric_history.update({f'val_{t}': [] for t in target_names})

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    if not temporal_sampling_epoch:
        ts_idx = np.random.randint(0,ts_range  - seq_length, temporal_sampling_size)
    
    for epoch in tqdm(range(num_epochs)):
        current_lr = get_lr(opt)
        
        print(f"Epoch {epoch}/{num_epochs - 1}, current lr={current_lr}")
        
        # every epoch generate a new set of random time indices (sampling the timeseries)
        if temporal_sampling_epoch:
            ts_idx = np.random.randint(0, ts_range  - seq_length, temporal_sampling_size)

    
        model.train()
        train_loss, train_metric = loss_epoch(
            model, loss_func, metric_func, train_dl, target_names, device, opt, ts_idx, seq_length
        )

        loss_history["train"].append(train_loss)
        for t in target_names: metric_history[f'train_{t}'].append(train_metric[t])

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(
                model, loss_func, metric_func, val_dl, target_names, device, ts_idx= ts_idx, seq_length = seq_length
            )

        loss_history["val"].append(val_loss)
        for t in target_names: metric_history[f'val_{t}'].append(val_metric[t])

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")

        lr_scheduler.step(val_loss)

        # lr_scheduler.step()
        #if current_lr != get_lr(opt):
        #    print("Loading best model weights!")
        #    model.load_state_dict(best_model_wts)

        print(f"train loss: {train_loss}, train metric: {train_metric}")
        print(f"val loss: {val_loss}, val metric: {val_metric}")
        print("-" * 10)
        


    model.load_state_dict(best_model_wts)
    
    return model, loss_history, metric_history
