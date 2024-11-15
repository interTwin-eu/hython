

class ParamRescalerMixin:
    def rescale(self, param, *args):
        return self.rescaler.rescale(param, *args)