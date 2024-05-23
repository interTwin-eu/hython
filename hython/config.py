from pathlib import Path
import os
from omegaconf import OmegaConf

DEFAULT_CONFIG_DIR = Path(os.path.dirname(__file__)).parent / "config"

class Config:
    def __init__(self, config_file, config_dir = None):
        
        if config_dir is None:
            config_dir = DEFAULT_CONFIG_DIR
        
        self.conf = OmegaConf.load(f"{config_dir}/{config_file}")




