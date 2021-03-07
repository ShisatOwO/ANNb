import numpy as np

from annb.templates import *


class Linear(Actfunc):
    @staticmethod
    def forward(x: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        return x

    @staticmethod
    def backward(dinp: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        return 1


class Relu(Actfunc):
    @staticmethod
    def forward(x: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        out = np.copy(x)
        out[raw_lout <= 0] = 0
        return out


class Softmax(Actfunc):
    @staticmethod
    def forward(inp: np.array, reference: np.array = None, raw_lout: np.array = None) -> np.array:
        """Implementierung der Formel aus Neural Networks from Scratch in Python übernommen"""
        exp = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def backward(dinp: np.array, reference: np.array = None, raw_lout: np.array = None):
        """Implementierung der Formel aus Neural Networks from Scratch in Python übernommen"""
        x = np.argmax(reference, axis=1) if len(reference.shape) == 2 else reference
        out = dinp.copy()
        out[range(len(reference)), x] -= 1
        return out / len(reference)

