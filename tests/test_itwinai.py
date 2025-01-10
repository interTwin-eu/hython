import pytest
import torch
from pathlib import Path 
import subprocess

DIR_CONFIG = Path(__file__).parent.parent / "hython/itwinai"

def test_training():
    ret = subprocess.run(["itwinai", "exec-pipeline" , "--config", f"{str(DIR_CONFIG)}/training.yaml", "--pipe-key", "rnn_training_pipeline",
                     "-o", "epochs=2"])        
    ret.check_returncode()


def test_calibration():
    ret = subprocess.call(["itwinai", "exec-pipeline" , "--config", f"{str(DIR_CONFIG)}/calibration.yaml", "--pipe-key", "calibration_pipeline",
                     "-o", "epochs=2"])       
    
    ret.check_returncode()


def test_inference():
    pass