from __future__ import annotations
import pandas as pd
from typing import List


# region Zeitreihen-Features

def build_time_series(
        df: pd.DataFrame,
        id_col: str,
        date_col: str,
        y_col: str,
        freq: str = "D"
) -> pd.DataFrame:
    """
    Baut aus Rohdaten eine aggregierte Zeitreihe.

    Schritte:
    1) Relevante Spalten auswählen und auf Standardnamen ("date", "qty") umbenennen.
    2) Datum in echtes Datumsformat konvertieren (Ungültiges -> NaT).
    3) Nach Datum sortieren und fehlende Datumszeilen entfernen.
    4) Nach gewünschter Frequenz resamplen (z. B. 'D' = täglich, 'W' = wöchentlich) und summieren.

    Args:
        df: Eingabedaten (mind. date_col, y_col).
        id_col: Spalte mit der Objekt-ID (hier nicht verwendet, aber für Konsistenz der Signatur).
        date_col: Spalte mit Datum/Zeit.
        y_col: Zielgröße (z. B. Absatzmenge).
        freq: Resampling-Frequenz ('D', 'W', ...).

    Returns:
        Aggregierte Zeitreihe mit Spalten: ['date', 'qty'].
    """
    d = df[[date_col, y_col]].copy()  # nur benötigte Spalten
    d = d.rename(columns={date_col: "date", y_col: "qty"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce")  # robustes Parsing
    d = d.dropna(subset=["date"]).sort_values("date")  # ungültige Datumswerte raus
    d = d.set_index("date").resample(freq).sum().reset_index()  # Aggregation je Periode
    return d


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ergänzt einfache kalendarische Features für Zeitreihen-Modelle.

    Fügt hinzu:
      - weekday (0=Montag ... 6=Sonntag)
      - month (1..12), year (YYYY)
      - weekofyear (ISO-Kalenderwoche)
      - is_weekend (1 am Wochenende, sonst 0)
    Erwartet eine Spalte 'date' vom Typ datetime64.

    Args:
        df: Zeitreihen-DataFrame mit Spalte 'date'.

    Returns:
        Kopie von df mit zusätzlichen Feature-Spalten.
    """
    d = df.copy()
    d["weekday"] = d["date"].dt.weekday
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(int)
    d["is_weekend"] = d["weekday"].isin([5, 6]).astype(int)  # 5=Sa, 6=So
    return d


def add_lags_and_ma(df: pd.DataFrame, lags: List[int], windows: List[int]) -> pd.DataFrame:
    """
    Erzeugt Lag-Features und gleitende Durchschnitte für die Zielgröße 'qty'.

    Args:
        df: Zeitreihen-DataFrame mit Spalte 'qty'.
        lags: Liste von Verzögerungen (z. B. [1, 7, 28]).
        windows: Fenstergrößen für Moving Average (z. B. [7, 28]).

    Returns:
        Kopie mit zusätzlichen Spalten 'lag_{L}' und 'ma_{W}'.
    """
    d = df.copy()
    for L in lags:
        d[f"lag_{L}"] = d["qty"].shift(L)  # verschobene Zielwerte
    for W in windows:
        d[f"ma_{W}"] = d["qty"].rolling(window=W).mean()  # gleitender Mittelwert
    return d


# endregion


# region Kunden-Features / Aggregation

def build_customer_features(
        df: pd.DataFrame,
        customer_col: str,
        order_date_col: str,
        amount_col: str,
        returned_col: str = "returned"
) -> pd.DataFrame:
    """
    Aggregiert Bestell-/Kundendaten zu Kunden-Features (RFM-ähnlich).

    Berechnete Features:
      - orders: Anzahl eindeutiger Bestellungen (per 'order_id')
      - revenue: Summe der Umsätze
      - avg_basket: Durchschnittlicher Bestellwert
      - return_rate: Anteil Rücksendungen (wenn returned_col ∈ {0,1} oder bool)
      - days_since_last: Tage seit letzter Bestellung (gegen globales letztes Datum)

    Erwartet Spalten:
      customer_col, 'order_id', order_date_col, amount_col, optional returned_col.

    Args:
        df: Rohdaten auf Bestell-/Kundenebene.
        customer_col: Spalte mit Kunden-ID.
        order_date_col: Spalte mit Bestelldatum.
        amount_col: Umsatz-/Nettobetrag pro Bestellung.
        returned_col: Optional, Spalte mit Rücksende-Flag (Default 'returned').

    Returns:
        DataFrame mit je einer Zeile pro Kunde und o. g. Feature-Spalten.
    """
    d = df.copy()
    d[order_date_col] = pd.to_datetime(d[order_date_col], errors="coerce")
    d = d.dropna(subset=[order_date_col])  # nur valide Bestelldaten
    last_date = d[order_date_col].max()  # Referenz für Recency

    grp = d.groupby(customer_col)
    features = grp.agg(
        orders=('order_id', 'nunique'),  # eindeutige Bestellungen
        revenue=(amount_col, 'sum'),  # Gesamtumsatz
        avg_basket=(amount_col, 'mean'),  # Ø Bestellwert
        last_order=(order_date_col, 'max'),  # letztes Bestelldatum
        returns=(returned_col, 'mean')  # Rücksendequote (falls vorhanden)
    )
    features = features.rename(columns={"returns": "return_rate"})

    # Tage seit letzter Bestellung (Recency); float für nachfolgende Skalierung/Modelle
    features["days_since_last"] = (last_date - features["last_order"]).dt.days.astype(float)

    # letztes Datum nicht mehr nötig, deshalb entfernen
    features = features.drop(columns=["last_order"])

    # fehlende Werte als 0 auffüllen (z. B. wenn returned_col fehlt -> NaN)
    features = features.fillna(0.0)
    return features.reset_index()


# endregion


# region Ausreißerbehandlung

def iqr_clip(df: pd.DataFrame, col: str = "qty", k: float = 1.5) -> pd.DataFrame:
    """
    Clipt Ausreißer in Spalte `col` mithilfe der IQR-Regel (Tukey-Fences).

    Methode:
      - IQR = Q3 - Q1
      - Untere/obere Grenze = Q1 - k*IQR, Q3 + k*IQR
      - Werte außerhalb werden auf die Grenze gekappt (nicht entfernt)

    Args:
        df: Eingabedaten.
        col: Zielspalte, die geclippt werden soll (Default: 'qty').
        k: Multiplikator für die Fence-Breite (Default: 1.5; 3.0 für strengere Outlier).

    Returns:
        Kopie von df mit geclippten Werten in `col`. Wenn IQR<=0, Rückgabe unverändert.
    """
    d = df.copy()
    q1, q3 = d[col].quantile(0.25), d[col].quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:  # keine Streuung -> nichts clippen
        return d
    lower, upper = q1 - k * iqr, q3 + k * iqr
    d[col] = d[col].clip(lower=lower, upper=upper)
    return d

# endregion
