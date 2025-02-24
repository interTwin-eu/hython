import torch
from typing import Dict
import importlib
from pathlib import Path
from .cudnnLSTM import CudaLSTM
from .convLSTM import ConvLSTM
from .hybrid import Hybrid
from .transferNN import TransferNN

MODULE_MODELS = importlib.import_module("hython.models")

def get_model_class(model_name: str):
    model = getattr(MODULE_MODELS, model_name)    
    return model

def load_model(model_registry: dict, model_name, model = None):
    model_config = model_registry.get(model_name)
    model_uri = Path(model_config.get("model_uri"))
    logger = model_config.get("logger")
    
    if logger == "mlflow":
        import mlflow
        model = mlflow.pytorch.load_model(str(model_uri))

    elif logger == "local":
        from omegaconf import OmegaConf
        ext = model_uri.suffix
        if model is None:
            raise ValueError("For loading a model from a local source, model must be an instantiated model class")
        if ext == ".yaml" or ext == ".yml":
            config = OmegaConf.load(model_uri)
            
            model_uri = config.model_logger[model_name]["model_uri"]
            # model_uri = Path(config.work_dir) / f"{config.experiment_name}_{config.experiment_run}" / config.model_file_name
        elif ext == ".pt" or  ext == "pth":
            pass
        model.load_state_dict(torch.load(model_uri))
    return model

def log_model(model_registry: dict, model_name, model):
    model_config = model_registry.get(model_name)
    model_uri = Path(model_config.get("model_uri"))
    
    logger = model_config.get("logger")
    
    if logger == "mlflow":
        raise NotImplementedError
    if logger == "local":
        torch.save(model.state_dict(), model_uri)



class ModelLogAPI():
    def __init__(self, cfg):
        self.cfg = cfg.model_logger
        self.model_components = {config["model_component"]:model for model, config in self.cfg.items()}
        self.model_log_names = {config["model_component"]:config["model_name"] for model, config in self.cfg.items() if config["log"] is True}
        self.model_load_names = {config["model_component"]:config["model_name"] for model, config in self.cfg.items() if config["load"] is True}
        self.model_loggers = {config["model_component"]:config["logger"] for model, config in self.cfg.items()}
        
    def load_model(self, model_component, model_instantiated = None):
        model_class_name = self.model_components[model_component]
        return load_model(self.cfg, model_class_name, model = model_instantiated)
        
    def log_model(self, model_component, model):
        model_name = self.model_log_names.get(model_component)
        log_model(self.cfg, model_name, model)
        
    def get_model_component_class(self, model_component):
        return self.model_components[model_component]

    def get_model_log_name(self, model_component):
        return self.model_log_names[model_component]
        
    def get_model_log_names(self) -> Dict:
        return self.model_log_names

    def get_model_load_name(self, model_component) -> Dict:
        return self.model_load_names[model_component]

    def get_model_load_names(self) -> Dict:
        return self.model_load_names
    def get_model_logger(self, model_component):
        return self.model_loggers[model_component]
