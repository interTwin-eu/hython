import torch
import importlib
from .cudnnLSTM import CudaLSTM
from .hybrid import Hybrid
from .transferNN import TransferNN

MODULE_MODELS = importlib.import_module("hython.models")

def get_model_class(model_name: str):
    model = getattr(MODULE_MODELS, model_name)    
    return model

def load_model(model_registry: dict, model_name):
    if (mlflow_models := model_registry.get("mlflow")) is not None:
        import mlflow
        config = mlflow_models.get(model_name)
        model_uri = config.get("model_uri")
        model = mlflow.pytorch.load_model(model_uri)
    elif (local_models := model_registry.get("local")) is not None:
        config = local_models.get(model_name)
        model = get_model_class(model_name)
        model_fp = config.get("model_uri")
        model.load_state_dict(torch.load(model_fp))
    return model


from typing import Dict
class ModelLogAPI():

    def __init__(self, cfg):
        self.cfg = cfg.model_logger

        if (mlflow_logger := self.cfg.get("mlflow")) is not None:
            
            self.model_components = {mlflow_logger[model]["model_component"]:model for model in mlflow_logger } 
            self.model_log_names = {mlflow_logger[model]["model_component"]:mlflow_logger[model]["model_name"]
                                    for model in mlflow_logger if mlflow_logger[model]["log"] == True} 

    def load_model(self, model_component):
            model_class_name = self.model_components[model_component]
            return load_model(self.cfg, model_class_name)
        
    def get_model_component_class(self, model_component):
            return self.model_components[model_component]

    def get_model_log_name(self, model_component):
            return self.model_log_names[model_component]
        
    def get_model_log_names(self) -> Dict:
            return self.model_log_names