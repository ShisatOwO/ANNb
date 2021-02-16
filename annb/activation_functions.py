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
        return 1 if x < 0 else 0


class Sigmoid(Actfunc):
    @staticmethod
    def forward(x: np.array) -> np.array:
        return 1 / (1 + np.power(np.e, -x))


class Softmax(Actfunc):
    @staticmethod
    def forward(x: np.array) -> np.array:
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
