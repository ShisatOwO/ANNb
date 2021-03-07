import numpy as np

from abc import ABC, abstractmethod


class Optimizer(ABC):
    @staticmethod
    @abstractmethod
    def optimize(layer, learning_rate: float) -> None:
        """Optimiert gegebenen Layer"""
        ...

    @abstractmethod
    def __init__(self, learning_rate: float = None, decay: float = None, momentum: float = None) -> None:
        self._initial_lr = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._lr = learning_rate

    @abstractmethod
    def train(self, network, iterations: int, dataset: [np.array, np.array]) -> None:
        """Trainiert das Gesamte Netzwerk, basierend auf gegebenen Parametern"""
        ...


class Actfunc(ABC):
    @staticmethod
    @abstractmethod
    def forward(inp: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        """Führt einen Forward Pass durch die Aktivierungsfunktion durch"""
        ...

    @staticmethod
    def backward(dinp: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        """Führt einen Backward Pass durch die Aktivierungsfunktion durch"""
        return 1


class Layer(ABC):
    @abstractmethod
    def __init__(self, connections: int, neurons: int) -> None:
        self.out: np.array = None
        self.raw_out: np.array = None

        self._weights = np.random.randn(connections, neurons) / 100     # Zufällige Gewichte nah an Null Generieren
        self._bias = np.array = np.zeros([1, neurons])                  # Bias mit NUllen Initialisieren.
        self._inp = None
        self._dw = None
        self._db = None

    @abstractmethod
    def fpass(self, inp: np.array, actfunc: Actfunc) -> np.array:
        """Führt einen Forward Pass Durch."""
        ...

    @abstractmethod
    def bpass(self, dinp: np.array, actfunc: Actfunc, reference: np.array) -> np.array:
        """Führt einen Backward Pass Durch."""
        ...

    def optimize(self, optimizer: Optimizer, learning_rate: float) -> None:
        """Optimiert den Layer, es gibt noch keine Error checks ob vorher  wirklich ein Backward Pass durchgeführt wurde"""
        ...
