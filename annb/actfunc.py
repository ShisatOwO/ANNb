import numpy as np

from abc import ABC, abstractmethod


class Actfunc(ABC):
    @staticmethod
    @abstractmethod
    def forward(inp: np.array) -> np.array:
        ...

    @staticmethod
    def backward(dinp: np.array) -> np.array:
        return 1


class CCESoftmax(Actfunc):
    ...

class Relu(Actfunc):
    ...

class Linear(Actfunc):
    ...
