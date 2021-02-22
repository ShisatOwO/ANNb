import numpy as np

from annb.templates import *


class BatchSGD(Optimizer):
    def __init__(self, learning_rate: float = None, decay: float = None, momentum: float = None) -> None:
        super().__init__(learning_rate, decay, momentum)
        self.manager = None

    @staticmethod
    def optimize(layer: Layer, learning_rate: float) -> None:
        layer._weights -= learning_rate * layer._dw
        layer._bias -= learning_rate * layer._db

    def train(self, network, iterations: int, dataset: [np.array, np.array]) -> None:
        # TODO: Error checking (überall eigentlich)
        if self._initial_lr is not None and self._decay is not None:
            for layer in network:
                setattr(layer.layer, "w_mom", np.zeros_like(layer.layer._weights))
                setattr(layer.layer, "b_mom", np.zeros_like(layer.layer._bias))

            for i in range(iterations):

                self.manager.guess(dataset[0])
                self.manager.calc_gradient(dataset[1])

                self._lr = self._initial_lr * (1 / (1 + self._decay * i))

                for layer in network:
                    if self._momentum:
                        w_mom = self._momentum * layer.layer.w_mom - self._lr * layer.layer._dw
                        b_mom = self._momentum * layer.layer.b_mom - self._lr * layer.layer._db
                        layer.layer.w_mom = w_mom
                        layer.layer.b_mom = b_mom
                        layer.layer._weights += w_mom
                        layer.layer._bias += b_mom

                    else:
                        layer.layer.optimize(self, self._lr)

                #TODD: In eigene methode printer verschieben
                print(f"Iteration:\t\t{i}")
                print(f"LR:\t\t{self._lr}")
                print(f"Accuracy:\t\t{self.manager.calc_acc(dataset[1])}")
                print(f"Loss:\t\t{np.mean(self.manager.calc_loss(dataset[1]))}\n")




