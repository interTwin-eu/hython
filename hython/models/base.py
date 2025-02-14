import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, cfg = None):
        super(BaseModel, self).__init__()
        self.cfg = cfg

    def rescale_input(self, param):
        return param

    def rescale_output(self, data):
        return data