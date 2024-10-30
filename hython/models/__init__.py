

class ParamRescalerMixin:
    def rescale(self, param):
        return self.rescaler.rescale(param)