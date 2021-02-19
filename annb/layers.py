import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod

from annb.actfunc import Actfunc
from annb.optimizer import *


class Layer(ABC):
    @abstractmethod
    def __init__(self, connections: int, neurons: int) -> None:
        self.out: np.array = None

        self._weights = np.random.randn(connections, neurons) / 100

    @abstractmethod
    def fpass(self, inp: np.array, actfunc: Actfunc) -> np.array:
        ...

    def bpass(self, dinp: np.array, actfunc: Actfunc) -> None:
        ...

    def optimize(self, optimizer: Optimizer, gradient: [np.array, np.array]):
        ...


@dataclass
class LayerInfo:
    layer: Layer
    actfunc: Actfunc
    gradient: [np.array, np.array] = None
