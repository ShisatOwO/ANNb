import numpy as np

from annb.layers import *
from annb.func import *


class NetworkManager:
    _layers: [LayerInfo, ]
    out: np.array

    def __init__(self, lin: LayerInfo = None, lout: LayerInfo = None) -> None:
        self._lin = lin
        self._lout = lout
        self._lhidden: [LayerInfo, ] = None

    def _merge_layers(self) -> None:
        self._layers = [] if self._lhidden is None else self._lhidden
        self._layers.append(self._lout)
        self._layers.insert(0, self._lin)

    def add_lhidden(self, layer: LayerInfo) -> None:
        if self._lhidden is None:
            self._lhidden = []
        self._lhidden.append(layer)
        self._merge_layers()

    def guess(self, inp: np.array) -> None:
        self._merge_layers()

        self._layers[0].layer.pass_forward(inp, self._layers[0].act_func)
        for layer, prev_layer in zip(self._layers[1::], self._layers[:-1:]):
            layer.layer.pass_forward(prev_layer.layer.out, layer.act_func)

        self.out = self._layers[-1].layer.out

    def calc_loss(self, inp: np.array, goal: np.array) -> float:
        #TODO: Die Loss function kann eigentlich komplett hierrüber verschoben werden
        self.guess(inp)
        return LossFunction.cce(goal, self.out)

    def calc_acc(self, goal: np.array):
        inp = np.argmax(self.out, axis=1)
        return np.mean(inp == goal)

    def back_prop(self, loss: float):
        ...
