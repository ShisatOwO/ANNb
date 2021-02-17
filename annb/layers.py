import numpy as np

from dataclasses import dataclass

from annb.activation_functions import *


class Layer:
    _current_inp: np.array  # Speichert letzten input
    _weights: np.array      # Matrix (n dimensionaler array) für die vershiedenen gewichte der vershiedenen Neuronen
    _biases: np.array       # Array mit einem Bias für jedes Neuron
    out: np.array           # Speichert das Ergebnis dieses Layers nach einem Pass.
    raw_out: np.array       # Speichert das Ergebnis dieses Layers nach einam Pass ohne Aktivierungs funktion
    dweights: np.array      # Ableitung der Gewichte
    dbiases: np.array       # Ableitung der Biase
    dinp: np.array          # Ableitung des Inputs

    def pass_forward(self, inp: np.array, act_func: Actfunc) -> None:
        """Führt einen Forwardpass mit angebenem Input und Aktivierungsfunktion durch und speichert das Ergebnis in DenseLayer.out"""
        self.raw_out = np.dot(inp, self._weights) + self._biases
        self.out = act_func.forward(self.raw_out)
        self._current_inp = inp

    def pass_backward(self, prev_d: np.array, act_func: Actfunc, *args) -> None:
        dactf = Actfunc.backward(self.out)
        if isinstance(act_func, Softmax):
            dactf = act_func.cce_backward(args[0], prev_d)

        dactf *= self.raw_out

        self.dinp = np.dot(dactf, self._weights.T)
        self.dweights = np.dot(self._current_inp.T, dactf)
        self.dbiases = np.sum(dactf, axis=0, keepdims=1)

    #TODO: Überflüssig...
    def calc_loss(self, ideal: np.array, loss_func: callable) -> np.array:
        ...


class DenseLayer(Layer):
    def __init__(self, connections: int, neurons: int, scaling: int = 100) -> None:
        # Der Anzahl, der Neuronen und Verbindungen mit übergeordneten Layern, entsprechend zufällige Gewichte generieren.
        # Die zufälligen Gewichte sollten nah an Null liegen um das Training nicht zu beeinflussen.
        self._weights: np.array = np.random.randn(connections, neurons) / scaling

        # 1*neurons grosse Matrix mit Nullen als Bias. Null ist der beste ausgangs wert zum trainieren des Bias.
        self._biases: np.array = np.zeros([1, neurons])


@dataclass
class LayerInfo:
    layer: Layer
    act_func: Actfunc
