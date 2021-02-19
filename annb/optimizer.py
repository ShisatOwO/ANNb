import numpy as np

from abc import ABC, abstractmethod


class Optimizer(ABC):
    @staticmethod
    @abstractmethod
    def optimize() -> None:
        ...


class BatchSGD(Optimizer):
    ...
