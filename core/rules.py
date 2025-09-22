
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_ts_stats(df: pd.DataFrame, y_col: str = "qty") -> dict:
    """
    Berechnet einfache statistische Kennzahlen für eine Zeitreihe,
    um später eine Modell-Empfehlung ableiten zu können.
    """

    # Droppe Zeilen ohne Werte in der Zielspalte (qty)
    d = df.dropna(subset=[y_col]).copy()
    n = len(d)

    # Falls nach dem Droppen keine Werte übrig sind → leeres Ergebnis zurückgeben
    if n == 0:
        return {"n": 0}

    # Numpy-Array mit den Absätzen als float
    y = d[y_col].values.astype(float)

    # Anteil der Tage mit Absatz = 0 (Nulltage)
    # Hoher Wert → viele Nullabsätze, Daten sind "löchrig"
    null_share = float((y == 0).mean())

    # Varianz der Absatzwerte (als grobes Maß für Volatilität)
    var = float(np.var(y))

    # Hilfsfunktion: Autokorrelation bei gegebenem Lag
    def acf(x, lag):
        x_ = (x - x.mean())  # Mittelwertzentrierung
        # Korrelation zwischen Serie und ihrer lag-Version
        return float(np.corrcoef(x_[:-lag], x_[lag:])[0, 1]) if lag < len(x) else 0.0

    # Autokorrelation bei 7 Tagen (→ wöchentliche Muster)
    acf7 = acf(y, 7) if n > 7 else 0.0
    # Autokorrelation bei 30 Tagen (→ monatliche Muster)
    acf30 = acf(y, 30) if n > 30 else 0.0

    # Alles gesammelt als Dictionary zurückgeben
    return {"n": n, "null_share": null_share, "var": var, "acf7": acf7, "acf30": acf30}


def recommend_model(stats: dict) -> tuple[str, str]:
    """
    Leitet aus den Zeitreihen-Statistiken eine Modell-Empfehlung ab.
    Gibt (modellname, begründung) zurück.
    """

    # 1) Kurze Serie (< 60 Punkte):
    # Prophet kommt besser mit kleinen Datenmengen zurecht,
    # da er starke Annahmen über Trends/Saisonalität hat.
    if stats.get("n", 0) < 60:
        return "prophet", "Serie kurz → interpretiertes Modell (Prophet) stabiler."

    # 2) Wenig Nullwerte (< 20%) und klare Autokorrelation
    # bei 7 oder 30 Tagen (|acf| > 0.4):
    # → Serie hat saubere Saisonalität, Prophet kann das gut abbilden.
    if stats.get("null_share", 1) < 0.2 and (
        abs(stats.get("acf7", 0)) > 0.4 or abs(stats.get("acf30", 0)) > 0.4
    ):
        return "prophet", "Klare Saisonalität, wenige Ausreißer → Prophet."

    # 3) Standard-Fall:
    # Wenn weder kurz noch klare Saisonalität → eher volatil/komplex.
    # LSTM ist flexibler und kann nichtlineare Muster lernen.
    return "lstm", "Volatilität/Komplexität → LSTM kann nichtlineare Muster abbilden."