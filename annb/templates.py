import numpy as np

from abc import ABC, abstractmethod


class Optimizer(ABC):
    @staticmethod
    @abstractmethod
    def optimize(layer, learning_rate: float) -> None:
        ...


class Actfunc(ABC):
    @staticmethod
    @abstractmethod
    def forward(inp: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        ...

    @staticmethod
    def backward(dinp: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        return 1


class Layer(ABC):
    @abstractmethod
    def __init__(self, connections: int, neurons: int) -> None:
        self.out: np.array = None
        self.raw_out: np.array = None

        self._weights = np.random.randn(connections, neurons) / 100
        self._bias = np.array = np.zeros([1, neurons])
        self._inp = None
        self._dw = None
        self._db = None

    @abstractmethod
    def fpass(self, inp: np.array, actfunc: Actfunc) -> np.array:
        ...

    @abstractmethod
    def bpass(self, dinp: np.array, actfunc: Actfunc, reference: np.array) -> np.array:
        ...

    def optimize(self, optimizer: Optimizer, learning_rate: float) -> None:
        ...
