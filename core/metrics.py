
from __future__ import annotations
import numpy as np
from sklearn.metrics import silhouette_score

def rmse(y_true, y_pred) -> float:
    # Konvertiert Eingaben in NumPy-Arrays vom Typ float
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Berechnet Root Mean Squared Error (RMSE):
    # 1. Differenz (Fehler) quadrieren
    # 2. Mittelwert bilden
    # 3. Quadratwurzel ziehen
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true, y_pred) -> float:
    # Sicherstellen, dass Eingabe ein NumPy-Array ist
    y_true = np.asarray(y_true, dtype=float)

    # Wenn keine Daten vorliegen, Ergebnis = NaN
    if len(y_true) == 0:
        return float("nan")

    # Normierungsfaktor = Spannweite (max - min)
    denom = float(np.max(y_true) - np.min(y_true))

    # Wenn alle Werte gleich sind (Spannweite = 0) → Fehler immer 0
    if denom == 0:
        return 0.0

    # NRMSE = RMSE geteilt durch Spannweite
    return rmse(y_true, y_pred) / denom


def silhouette(X: np.ndarray, labels) -> float:
    # Abbruch, wenn zu wenige Daten oder nur 1 Cluster
    if X.shape[0] < 2 or len(set(labels)) < 2:
        return 0.0

    try:
        # Silhouette Score messen → je näher an 1, desto besser getrennt
        return float(silhouette_score(X, labels))
    except Exception:
        # Falls Fehler (z. B. numerisch instabil), Rückfall = 0.0
        return 0.0


def last_horizon_split(y, horizon: int):
    # Eingabe in NumPy-Array konvertieren
    y = np.asarray(y, dtype=float)

    # Wenn Horizon ≤ 0 oder Daten kürzer als Horizon → kein Split
    if horizon <= 0 or len(y) <= horizon:
        return y, np.asarray([], dtype=float)

    # Aufteilen in (Trainingsdaten, Testdaten):
    # Trainingsdaten = alles außer den letzten 'horizon'-Werten
    # Testdaten = die letzten 'horizon'-Werte
    return y[:-horizon], y[-horizon:]
