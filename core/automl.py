# core/automl.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

from . import metrics
from . import forecast_prophet, forecast_lstm
from . import cluster_kmeans

@dataclass
class CandidateResult:
    name: str
    params: Dict[str, Any]
    rmse: float
    nrmse: float
    success: bool
    error: Optional[str] = None
    extra: Dict[str, Any] = None  # place to store debug or timings


def _eval_prophet(ts_df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # nutze interne Prophet-Pipeline; params sind aktuell nur „durchgereichte“ Hinweise
    holidays_path = params.get("holidays_path")
    fc_df, m = forecast_prophet.fit_predict(ts_df, horizon=horizon, holidays_path=holidays_path)
    return fc_df, m


def _eval_lstm(ts_df: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    fc_df, m = forecast_lstm.fit_predict(
        ts_df,
        horizon=int(horizon),
        window=int(params.get("window", 30)),
        epochs=int(params.get("epochs", 30)),
    )
    return fc_df, m


def select_and_tune_forecast(
    ts_df: pd.DataFrame,
    horizon: int,
    search_space: Dict[str, List[Dict[str, Any]]],
    holidays_path=None,
) -> Tuple[str, Dict[str, Any], pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Testet mehrere Kandidaten (Prophet/LSTM) auf einem Holdout (siehe forecast_* Pipelines)
    und gibt bestes Modell + Forecast + Metriken + Leaderboard zurück.

    search_space Beispiel:
    {
      "prophet": [{"holidays": True}, {"holidays": False}],
      "lstm": [{"window": 24, "epochs": 15}, {"window": 30, "epochs": 20}]
    }
    """
    leaderboard: List[CandidateResult] = []
    best = None  # (name, params, forecast_df, metrics)

    # Prophet-Kandidaten
    for p in search_space.get("prophet", []):
        params = {"holidays_path": holidays_path if p.get("holidays", True) else None}
        try:
            fc, m = _eval_prophet(ts_df, horizon, params)
            res = CandidateResult("prophet", p, float(m["rmse"]), float(m["nrmse"]), True)
            leaderboard.append(res)
            if best is None or (res.rmse == res.rmse and res.rmse < best[3]["rmse"]):
                best = ("prophet", p, fc, m)
        except Exception as e:
            leaderboard.append(CandidateResult("prophet", p, float("inf"), float("inf"), False, error=str(e)))

    # LSTM-Kandidaten
    for p in search_space.get("lstm", []):
        try:
            fc, m = _eval_lstm(ts_df, horizon, p)
            res = CandidateResult("lstm", p, float(m["rmse"]), float(m["nrmse"]), True)
            leaderboard.append(res)
            if best is None or (res.rmse == res.rmse and res.rmse < best[3]["rmse"]):
                best = ("lstm", p, fc, m)
        except Exception as e:
            leaderboard.append(CandidateResult("lstm", p, float("inf"), float("inf"), False, error=str(e)))

    if best is None:
        raise RuntimeError("Keine AutoML-Kandidaten erfolgreich — prüfe Datenlänge/Packages.")

    # Leaderboard als DataFrame
    lb_rows = []
    for r in leaderboard:
        row = {
            "model": r.name,
            "params": r.params,
            "rmse": r.rmse,
            "nrmse": r.nrmse,
            "success": r.success,
            "error": r.error,
        }
        lb_rows.append(row)
    lb_df = pd.DataFrame(lb_rows).sort_values(by=["success", "rmse"], ascending=[False, True])

    return best[0], best[1], best[2], best[3], lb_df


def select_k_for_clustering(df_customer_features: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> Tuple[int, pd.DataFrame]:
    """Wrapper für Silhouette-basiertes k-Scoring (mit Tabelle für UI)."""
    tbl = cluster_kmeans.silhouette_for_k(df_customer_features, range(k_min, k_max + 1))
    if tbl.empty:
        return max(2, min(3, len(df_customer_features))), tbl
    best_row = tbl.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best_row["k"]), tbl
