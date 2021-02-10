import numpy as np


class ActivationFunction(object):
    """Verschiedene Aktivierungsfunktionen"""
    @staticmethod
    def linear(x: np.array) -> np.array:
        return x

    @staticmethod
    def relu(x: np.array) -> np.array:
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1/(1+np.power(np.e, -x))

    @staticmethod
    def softmax(x: np.array) -> np.array:
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)


class LossFunction(object):
    @staticmethod
    def cce(goal: np.array, prediction: np.array) -> np.array:
        # Es könnte vorkommen, dass prediction eine Null enthält. Deshalb werden werden alle Werte, die kleiner als 10^-9,
        # Auf ein Milliardstel gesetzt (10^-9). Wenn ein wert Null wäre gabe es ein Fehler bei der Log Funktion.
        prediction = np.clip(prediction, 1e-9, 1)

        # Die Dimensionen von goal erweitern, damit es zu den dimensionen der Prediction passt.
        ggoal = []
        for g in goal:
            t = [0] * (prediction.shape[1] - 1)
            t.insert(g, 1)
            ggoal.append(t)

        return -np.sum(np.log(prediction) * ggoal, axis=1)
