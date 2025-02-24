from torch.nn import functional as F
import copy
import importlib.util


tqdm_support = True if importlib.util.find_spec("tqdm") is not None else False

if tqdm_support:
    from tqdm.auto import tqdm


class BaseTrainParams:
    pass


from .base import *

from .train import train_val
from .rnn import RNNTrainer
from .conv import ConvTrainer
from .cal import CalTrainer


TRAINERS = {
    "ConvTrainer":ConvTrainer,
    "RNNTrainer":RNNTrainer,
    "CalTrainer":CalTrainer,
}

DEPRECATED = []

def get_trainer(trainer):
    if trainer in DEPRECATED:
        VALID = set(TRAINERS.keys()).difference(set(DEPRECATED))
        raise DeprecationWarning(f"dataset {trainer} is deprecated, available datasets {VALID}")
    return TRAINERS.get(trainer)