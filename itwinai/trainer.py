from typing import Dict, Any
import logging
from itwinai.components import Trainer, monitor_exec
from itwinai.loggers import WanDBLogger
import matplotlib.pyplot as plt

from hython.models.lstm import CustomLSTM
from hython.train_val import train_val

from hython.losses import RMSELoss
from hython.metrics import mse_metric

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from importlib import import_module

class LSTMTrainer(Trainer):
    def __init__(
        self,
        model: dict,
        dynamic_names: list, 
        static_names : list, 
        target_names : list, 
        spatial_batch_size: int = None, 
        temporal_sampling_size: int = None, 
        seq_length : int = None, 
        hidden_size: int = None,  
        input_size : int = None,  
        path2models: str = None, 
        epochs: int = None,
        wandb_project: str = "test_project",
        wandb_run: str = "test"
    ):
        super().__init__()
        
        self.save_parameters(**self.locals2params(locals()))
        self.model = model
        self.spatial_batch_size     = spatial_batch_size
        self.temporal_sampling_size = temporal_sampling_size
        self.target_names           = target_names
        self.static_names           = static_names
        self.path2models            = path2models
        self.epochs                 = epochs
        self.seq_length             = seq_length
        self.wandb_project          = wandb_project
        self.wandb_run              = wandb_run

        self.model_params={
            "input_size": input_size, 
            "hidden_size": hidden_size, 
            "output_size": len(target_names), 
            "number_static_predictors": len(static_names),
            "target_names": target_names, 
        }

        

    @monitor_exec
    def execute(self, dataset, train_sampler, valid_sampler ) -> None:
        
        #wandb
        wandb = WanDBLogger() #**dict(project=self.wandb_project, name = self.wandb_run))

        #setup device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.debug("Info:: Device set to", device)

        #setup data
        train_loader = DataLoader(dataset, batch_size=self.spatial_batch_size, shuffle=False, sampler = train_sampler) # implement shuffling in the sampler!
        val_loader = DataLoader(dataset, batch_size=self.spatial_batch_size, shuffle=False, sampler = valid_sampler)  
                
        logging.debug("Info: Data loaded to torch")  

        #model
        #model = CustomLSTM(self.model_params)
        model_cls = getattr(import_module(self.model.get("module_path")), self.model.get("class"))
        model = model_cls(**self.model.get("init_args"))
        model = model.to(device)
        print(model)

        #optimizer
        opt = optim.Adam(model.parameters(), lr=1e-3)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=10)

        #loss
        loss_fn = RMSELoss(target_weight={"actevap":0.5, "vwc":0.5})

        ## Set the training parameters
        params_train={
            "num_epochs": self.epochs,
            "temporal_sampling_idx_change_with_epoch": True,
            "temporal_sampling_size": self.temporal_sampling_size,
            "seq_length": self.seq_length,
            "ts_range": dataset.y.shape[1],
            "optimizer": opt,
            "loss_func": loss_fn,
            "metric_func": mse_metric,
            "train_dl": train_loader, 
            "val_dl"  : val_loader,
            "lr_scheduler": lr_scheduler,
            "path2weights": f"{self.path2models}/weights.pt", 
            "device": device,
            "target_names": self.target_names
        }
        logging.debug("Info: Model compiled")
        wandb.save_hyperparameters(params_train)

        #train
        model, sm_loss_history , sm_metric_history = train_val(model, params_train, wandb)      
        logging.debug("Info:: Model trained")

        lepochs = list(range(1,params_train["num_epochs"] + 1))

        fig, axs = plt.subplots(3, 1, figsize= (12,6), sharex=True)

        axs[0].plot(lepochs, sm_metric_history['train_vwc'], marker='.', linestyle='-', color='b', label='Training')
        axs[0].plot(lepochs, sm_metric_history['val_vwc'], marker='.', linestyle='-', color='r', label='Validation')
        #axs[0].title('Validation Loss - SM')
        axs[0].grid(True)
        axs[0].legend(bbox_to_anchor=(1,1))

        axs[1].plot(lepochs, sm_metric_history['train_actevap'], marker='.', linestyle='-', color='b', label='Training')
        axs[1].plot(lepochs, sm_metric_history['val_actevap'], marker='.', linestyle='-', color='r', label='Validation')
        #axs[0].title('Validation Loss - ET')
        axs[1].grid(True)

        axs[2].plot(lepochs, [i.detach().cpu().numpy() for i in sm_loss_history['train']], marker='.', linestyle='-', color='b', label='Training')
        axs[2].plot(lepochs, [i.detach().cpu().numpy() for i in sm_loss_history['val']], marker='.', linestyle='-', color='r', label='Validation')
        #axs[0].title('Validation Loss - Combined')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel(loss_fn.__name__)
        axs[2].grid(True)
        fig.savefig("loss.png")

    def load_state(self):
        return super().load_state()

    def save_state(self):
        return super().save_state()
