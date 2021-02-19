import numpy as np
from annb.layers import *


class NManager:
    def __init__(self, in_layer: LayerInfo, out_layer: LayerInfo) -> None:
        self.out: np.array = None

        self._layers = [in_layer, out_layer]

    def add_layer(self, layer: LayerInfo) -> None:
        self._layers.insert(-1, layer)

    def guess(self, input: np.array) -> np.array:
        inp = input
        for layer in self._layers:
            inp = layer.layer.fpass(inp, layer.actfunc)
        self.out = inp
        return inp

    def optimize(self) -> None:
        ...

    def calc_loss(self, reference: np.array) -> float:
        ...

    def calc_acc(self, reference: np.array) -> float:
        ...

    def calc_gradient(self, reference: np.array) -> None:
        ...
