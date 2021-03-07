import numpy as np

from dataclasses import dataclass

from annb.actfunc import Actfunc
from annb.optimizer import *
from annb.templates import *


class DenseLayer(Layer):
    def __init__(self, connections: int, neurons: int) -> None:
        """Initialisiert Layer mit angegebener Anzahl an Neuronen und legt die Angegebene Anzahl an Gewichten an."""
        super().__init__(connections, neurons)

    def fpass(self, inp: np.array, actfunc: Actfunc) -> np.array:
        """Führt einen Forward Pass Durch."""
        self._inp = inp                                         # Aktueller Input wird gespeichert
        self.raw_out = np.dot(inp, self._weights) + self._bias  # Zwischenergebnis vor der Aktivierungs Funktion speichern
        self.out = actfunc.forward(self.raw_out)                # Zwischenergebnis in Aktivierungsfunktion Einsetzen
        return self.out

    def bpass(self, dinp: np.array, actfunc: Actfunc, reference: np.array) -> np.array:
        """Führt einen Backward Pass durch"""
        din = actfunc.backward(dinp, reference, self.raw_out)   # Backward Pass durch die Aktivierungsfunktion (Ableitung)
        self._dw = np.dot(self._inp.T, din)                     # Einfluss der Gewichte berechenen
        self._db = np.sum(din, axis=0, keepdims=1)              # Einfluss der Biases berechenen
        self._inp = np.dot(din, self._weights.T)                # Alten Input ableiten.
        return self._inp

    def optimize(self, optimizer: Optimizer, learning_rate: float) -> None:
        """Optimiert diesen Layer"""
        optimizer.optimize(self, learning_rate)


# Daten Klasse zum speichern von information über diesen Layer.
@dataclass
class LayerInfo:
    layer: Layer
    actfunc: Actfunc
