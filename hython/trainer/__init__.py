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
from .rnn import *
from .conv import *
from .cal import *
