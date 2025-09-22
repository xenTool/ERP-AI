
from __future__ import annotations
import pandas as pd
from pathlib import Path
from . import metrics

def _load_holidays(path: Path | None) -> pd.DataFrame | None:
    # Nur versuchen, wenn ein Pfad übergeben wurde und die Datei existiert
    if path and Path(path).exists():
        try:
            # CSV einlesen (erwartet Spalten "ds" und "holiday")
            df = pd.read_csv(path)
            # Prüfen, ob die Mindestspalten vorhanden sind
            if {"ds", "holiday"}.issubset(df.columns):
                # "ds" (Datums-Spalte) in echtes Datum konvertieren
                df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
                # Zeilen ohne gültiges Datum verwerfen und DataFrame zurückgeben
                return df.dropna(subset=["ds"])
        except Exception:
            # Robustheits-Fallback: stillschweigend None zurückgeben,
            # wenn etwas beim Einlesen/Parsen schiefgeht
            pass
    # Kein Pfad, Datei fehlt oder ungültig → keine Holidays
    return None

def fit_predict(ts_df: pd.DataFrame, horizon: int = 30, holidays_path: Path | None = None):
    # Prophet importieren; wenn nicht installiert, mit Fehlermeldung abbrechen
    try:
        from prophet import Prophet
    except Exception as e:
        raise RuntimeError("Prophet ist nicht installiert. Bitte optional nachrüsten: pip install prophet") from e

    # Prophet erwartet Spaltennamen "ds" (Datum) und "y" (Ziel)
    df = ts_df[["date", "qty"]].rename(columns={"date": "ds", "qty": "y"}).copy()
    # Ungültige Zeilen entfernen und chronologisch sortieren
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")

    # Feiertage (optional) laden
    holidays = _load_holidays(holidays_path)

    # Prophet-Modell: wöchentliche + jährliche Saisonalität aktivieren;
    # (tägliche Saisonalität ist bei Tagesdaten oft nicht nötig)
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, holidays=holidays)
    # Modell fitten
    m.fit(df)

    # Zukunfts-DataFrame (nur Zukunft, ohne Historie) für 'horizon' Tage erzeugen
    future = m.make_future_dataframe(periods=int(horizon), freq="D", include_history=False)
    # Vorhersagen für die Zukunft berechnen
    fc = m.predict(future)

    # Output-DataFrame standardisieren (mit Konfidenzintervallen, wenn vorhanden)
    out = pd.DataFrame({
        "date": fc["ds"],
        "yhat": fc["yhat"],
        "yhat_lower": fc.get("yhat_lower"),
        "yhat_upper": fc.get("yhat_upper"),
    })

    # --- Backtest auf dem letzten 'horizon'-Fenster ---
    y = df["y"].values
    y_tr, y_te = metrics.last_horizon_split(y, int(horizon))
    # Wenn es keinen Testteil gibt (Serie zu kurz), Metriken als NaN zurückgeben
    if len(y_te) == 0:
        return out, {"rmse": float("nan"), "nrmse": float("nan")}

    # Trainingsschnitt (ohne die letzten 'h' Punkte) für Backtest-Fit
    df_tr = df.iloc[:len(y_tr)]

    # Zweites Prophet-Modell nur auf dem Trainingsschnitt fitten
    # (damit die Backtest-Metriken fair sind)
    m_bt = Prophet(weekly_seasonality=True, yearly_seasonality=True, holidays=holidays)
    m_bt.fit(df_tr)

    # Zukunfts-DF exakt in Testlänge erzeugen (ohne Historie)
    fut_te = m_bt.make_future_dataframe(periods=len(y_te), freq="D", include_history=False)
    # Testvorhersagen (yhat) extrahieren
    pred_te = m_bt.predict(fut_te)["yhat"].values

    # Fehlermaße (RMSE/NRMSE) zwischen echten Testwerten und Backtest-Predictions
    rm = metrics.rmse(y_te, pred_te)
    nrm = metrics.nrmse(y_te, pred_te)

    # Zukunftsforecast + Backtest-Metriken zurückgeben
    return out, {"rmse": rm, "nrmse": nrm}

