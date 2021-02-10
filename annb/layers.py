import numpy as np
from dataclasses import dataclass


class Layer(object):
    _weights: np.array     # Matrix (n dimensionaler array) für die vershiedenen gewichte der vershiedenen Neuronen
    _biases: np.array      # Array mit einem Bias für jedes Neuron
    out: np.array          # Speichert das Ergebnis dieses Layers nach einem Pass.

    def pass_forward(self, inp: np.array, act_func: callable) -> None:
        """Führt einen Forwardpass mit angebenem Input und Aktivierungsfunktion durch und speichert das Ergebnis in DenseLayer.out"""
        self.out = act_func(np.dot(inp, self._weights) + self._biases)

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
    act_func: callable
