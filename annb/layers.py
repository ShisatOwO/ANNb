import numpy as np

from dataclasses import dataclass

from annb.actfunc import Actfunc
from annb.optimizer import *
from annb.templates import *


class DenseLayer(Layer):
    def __init__(self, connections: int, neurons: int) -> None:
        super().__init__(connections, neurons)

    def fpass(self, inp: np.array, actfunc: Actfunc) -> np.array:
        self._inp = inp
        self.raw_out = np.dot(inp, self._weights) + self._bias
        self.out = actfunc.forward(self.raw_out)
        return self.out

    def bpass(self, dinp: np.array, actfunc: Actfunc, reference: np.array) -> np.array:
        din = actfunc.backward(dinp, reference, self.raw_out)
        self._dw = np.dot(self._inp.T, din)
        self._db = np.sum(din, axis=0, keepdims=1)
        self._inp = np.dot(din, self._weights.T)
        return self._inp

    def optimize(self, optimizer: Optimizer, learning_rate: float) -> None:
        optimizer.optimize(self, learning_rate)


@dataclass
class LayerInfo:
    layer: Layer
    actfunc: Actfunc
