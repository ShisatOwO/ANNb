import numpy as np

from abc import ABC, abstractmethod


class Actfunc(ABC):
    @staticmethod
    @abstractmethod
    def forward(x: np.array) -> np.array:
        ...

    @staticmethod
    def backward(x: np.array) -> np.array:
        return 1


class Linear(Actfunc):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return x


class Relu(Actfunc):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.array) -> np.array:
        out = np.zeros_like(x)
        out[x > 0] = 1
        return out


class Sigmoid(Actfunc):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return 1 / (1 + np.power(np.e, -x))


class Softmax(Actfunc):
    @staticmethod
    def forward(x: np.array) -> np.array:
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def cce_backward(x: np.array, dval: np.array) -> np.array:
        x = np.argmax(x, axis=1) if len(x.shape) == 2 else x
        out = dval.copy()
        out[range(len(dval)), x] -= 1
        return out / len(dval)
