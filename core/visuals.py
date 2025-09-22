from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import io

def plot_forecast(ts_df: pd.DataFrame, fc_df: pd.DataFrame, y_col: str = "qty"):
    """Zeitreihe + Prognose (inkl. Konfidenzintervall) plotten."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_df["date"], ts_df[y_col], label="Historie")  # Originaldaten
    if {"yhat"}.issubset(fc_df.columns):                    # Prognosewerte
        ax.plot(fc_df["date"], fc_df["yhat"], label="Prognose")
    if {"yhat_lower", "yhat_upper"}.issubset(fc_df.columns): # Unsicherheitsband
        ax.fill_between(fc_df["date"], fc_df["yhat_lower"], fc_df["yhat_upper"], alpha=0.2, label="Konfidenz")
    ax.set_xlabel("Datum")
    ax.set_ylabel(y_col)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_clusters_2d(plot_df: pd.DataFrame):
    """2D-Scatterplot von Clustern (z. B. K-Means) erstellen."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if "cluster" in plot_df.columns:                        # Wenn Clusterlabels vorhanden
        for cl in sorted(plot_df["cluster"].unique()):
            sub = plot_df[plot_df["cluster"] == cl]
            ax.scatter(sub["x"], sub["y"], s=18, label=f"Cluster {cl}")
        ax.legend()
    else:                                                   # Ohne Clusterlabels
        ax.scatter(plot_df["x"], plot_df["y"], s=18)
    ax.set_xlabel(plot_df.get("x_label", pd.Series(["x"]).iloc[0]))
    ax.set_ylabel(plot_df.get("y_label", pd.Series(["y"]).iloc[0]))
    ax.set_title("K-Means Cluster (2D Projektion)")
    fig.tight_layout()
    return fig

def plot_k_curves(tbl: pd.DataFrame):
    """Kurven für Silhouette und Inertia zur k-Auswahl plotten."""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(tbl["k"], tbl["silhouette"], marker="o")       # Silhouette-Scores
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette")
    ax1.set_title("k-Auswahl: Silhouette (und Inertia)")
    ax2 = ax1.twinx()                                       # zweite y-Achse
    ax2.plot(tbl["k"], tbl["inertia"], marker="x", alpha=0.5) # Inertia-Werte
    ax2.set_ylabel("Inertia")
    fig.tight_layout()
    return fig

def save_current_fig_png(fig, out_path: Path | None = None) -> tuple[bytes, str]:
    """Figure als PNG exportieren (Bytes + optional Datei)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)  # in Memory speichern
    buf.seek(0)
    name = "report_snapshot.png" if out_path is None else out_path.name
    if out_path is not None:                                     # optional auf Platte speichern
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(buf.getbuffer())
    return buf.getvalue(), name  # Rückgabe: PNG-Daten und Dateiname
