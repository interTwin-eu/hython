import numpy as np
import torch
from torch.nn.modules.loss import _Loss

import importlib.util
from abc import ABC
import copy

from hython.metrics import Metric
from hython.losses import PhysicsLossCollection
from .train import metric_epoch, loss_batch, train_val

tqdm_support = True if importlib.util.find_spec("tqdm") is not None else False

if tqdm_support:
    from tqdm.auto import tqdm


class BaseTrainParams:
    pass


# TODO: consider dataclass
class RNNTrainParams(BaseTrainParams):
    def __init__(
        self,
        loss_func: _Loss,
        metric_func: Metric,
        target_names: list,
        loss_physics_collection: PhysicsLossCollection = PhysicsLossCollection(),
        experiment: str = None,
        temporal_subsampling: bool = None,
        temporal_subset: list = None,
        seq_length: int = None,
        gradient_clip: dict = None,
    ):
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.loss_physics_collection = loss_physics_collection
        self.experiment = experiment
        self.temporal_subsampling = temporal_subsampling
        self.temporal_subset = temporal_subset
        self.seq_length = seq_length
        self.target_names = target_names
        self.gradient_clip = gradient_clip


class AbstractTrainer(ABC):
    def __init__(self, experiment: str):
        self.exp = experiment

    def temporal_index(self, args):
        pass

    def epoch_step(self):
        pass

    def predict_step(self):
        pass

    def save_weights(self, model, fp, onnx=False):
        if onnx:
            raise NotImplementedError()
        else:
            print(f"save weights to: {fp}")
            torch.save(model.state_dict(), fp)


from .rnn import *
from .conv import *
from .lumped import *
