from collections import UserDict


class Config(UserDict):
    def __getitem__(self, i):
        return vars(self)[i]