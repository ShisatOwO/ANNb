import numpy as np

from annb.templates import *


class BatchSGD(Optimizer):
    @staticmethod
    def optimize(layer: Layer, learning_rate: float) -> None:
        layer._weights -= learning_rate * layer._dw
        layer._bias -= learning_rate * layer._db

