import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def train_val(trainer, model, epochs, optimizer, lr_scheduler, dp_weights, device):

    target_names = trainer.P.target_names 
    
    loss_history = {"train": [], "val": []}    
    metric_history = {f'train_{target}': [] for target in target_names}
    metric_history.update({f'val_{target}': [] for target in target_names})
    
    best_loss = float("inf")
    
    for epoch in tqdm(range(epochs)):
        
        trainer.temporal_index() # For RNNs based models

        model.train()
        
        train_loss, train_metric = trainer.epoch_step(model, device, opt = optimizer)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = trainer.epoch_step(model, device, opt= None)
        
        lr_scheduler.step(val_loss)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
 
        for target in target_names: 
            metric_history[f'train_{target}'].append(train_metric[target])
            metric_history[f'val_{target}'].append(val_metric[target])

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            trainer.save_weights(model, dp_weights)
            print("Copied best model weights!")

        print(f"train loss: {train_loss}")
        print(f"val loss: {val_loss}")

    model.load_state_dict(best_model_weights)
            
    return model, loss_history, metric_history