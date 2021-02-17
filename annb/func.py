import numpy as np

#TODO: Es ist unsinnig das hier in eine eigene Klasse zu packen


class LossFunction:
    #TODO: Kann nach network_manager.py verschoben werden
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

    @staticmethod
    def reverse_cce(goal: np.array, d: np.array):
        if len(goal.shape) == 1:
            goal = np.eye(len(d[0]))[goal]
        out = -goal / d
        return out / len(d)

    @staticmethod
    def cost(goal: np.array, prediction: np.array) -> np.array:
        # Die Dimensionen von goal erweitern, damit es zu den dimensionen der Prediction passt.
        ggoal = []
        for g in goal:
            t = [0] * (prediction.shape[1] - 1)
            t.insert(g, 1)
            ggoal.append(t)

        return np.sum((np.array(ggoal) - prediction)**2)

