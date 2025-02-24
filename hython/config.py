


class Config(dict):
    def __getitem__(self, i):
        return vars(self)[i]