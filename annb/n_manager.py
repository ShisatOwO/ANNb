import numpy as np
from annb.layers import *


class NManager:
    def __init__(self, in_layer: LayerInfo, out_layer: LayerInfo) -> None:
        self.out: np.array = None

        self._layers = [in_layer, out_layer]
        self._network_optimizer = None

    def add_layer(self, layer: LayerInfo) -> None:
        self._layers.insert(-1, layer)

    def add_trainer(self, optimizer: Optimizer) -> None:
        self._network_optimizer = optimizer
        self._network_optimizer.manager = self

    def guess(self, input: np.array) -> np.array:
        inp = input
        for layer in self._layers:
            inp = layer.layer.fpass(inp, layer.actfunc)
        self.out = inp
        return inp

    def optimize(self, optimizer: Optimizer, learning_rate: float = 1, decay: float = 0, iteration: int = 0) -> None:
        for layer in self._layers:
            layer.layer.optimize(optimizer, learning_rate)

    def train(self, iterations: int, dataset: np.array, reference: np.array) -> None:
        if self._network_optimizer is not None:
            self._network_optimizer.train(self._layers, iterations, (dataset, reference))

    def calc_loss(self, reference: np.array) -> float:
        # Es könnte vorkommen, dass prediction eine Null enthält. Deshalb werden werden alle Werte, die kleiner als 10^-9,
        # Auf ein Milliardstel gesetzt (10^-9). Wenn ein wert Null wäre gabe es ein Fehler bei der Log Funktion.
        prediction = np.clip(self.out, 1e-9, 1-1e-9)

        # Die Dimensionen von goal erweitern, damit es zu den dimensionen der Prediction passt.
        ggoal = []
        for ref in reference:
            t = [0] * (prediction.shape[1] - 1)
            t.insert(ref, 1)
            ggoal.append(t)
        loss = -np.sum(np.log(prediction) * ggoal, axis=1)
        return loss

    def calc_acc(self, reference: np.array) -> float:
        pred = np.argmax(self.out, axis=1)
        x = np.argmax(reference, axis=1) if len(reference.shape) == 2 else reference
        return np.mean(reference==pred)

    def calc_gradient(self, reference: np.array) -> float:
        dinp = self.out
        for layer in self._layers[::-1]:
            dinp = layer.layer.bpass(dinp, layer.actfunc, reference)

        return np.mean(self.calc_loss(reference))
