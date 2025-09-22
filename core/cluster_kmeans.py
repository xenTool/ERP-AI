
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import metrics

def _scale_features(df_customer_features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, List[str]]:
    # Wählt nur numerische Spalten (int/float) aus, denn nur die sind für KMeans geeignet.
    feats = df_customer_features.select_dtypes(include=[float, int]).copy()

    # Liste der Spaltennamen merken (für spätere Interpretation)
    cols = feats.columns.tolist()

    # Standardisierung: Mittelwert = 0, Standardabweichung = 1.
    # Wichtig, weil KMeans sensibel auf Skalenunterschiede reagiert
    # (z. B. Umsatz in € vs. Bestellhäufigkeit als kleine Zahl).
    scaler = StandardScaler()

    # Transformation: Skaliert die Werte auf Numpy-Array X
    X = scaler.fit_transform(feats.values)

    # Gibt zurück: (skaliertes Array, den Scaler selbst, die Spaltennamen)
    return X, scaler, cols

def silhouette_for_k(df_customer_features: pd.DataFrame, k_range: range) -> pd.DataFrame:
    # Skaliert die Features
    X, _, _ = _scale_features(df_customer_features)

    rows = []
    # Testet für jedes k in der angegebenen Range
    for k in k_range:
        # Gültige Clusteranzahl: mindestens 2, maximal len(X)-1
        if k < 2 or k >= len(X):
            continue

        # KMeans mit k Clustern; n_init=10 = 10 verschiedene Randomstarts → stabiler
        km = KMeans(n_clusters=int(k), n_init=10, random_state=42)

        # Clustert die Daten und liefert Clusterlabels
        labels = km.fit_predict(X)

        # Berechnet den Silhouette Score (Maß für Clusterqualität)
        sil = metrics.silhouette(X, labels)

        # Speichert k, Silhouette und Inertia (Summe der Abstände innerhalb Cluster)
        rows.append({"k": int(k), "silhouette": sil, "inertia": float(km.inertia_)})

    # Baut DataFrame aus allen getesteten k-Werten, sortiert nach k
    return pd.DataFrame(rows).sort_values("k")

def best_k_by_silhouette(df_customer_features: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> int:
    # Berechnet Tabelle aller Kandidaten (Silhouette/Inertia je k)
    tbl = silhouette_for_k(df_customer_features, range(k_min, k_max + 1))

    # Falls keine gültigen Ergebnisse (zu wenige Daten)
    if tbl.empty:
        # Rückfall: mindestens 2, maximal 3 Cluster
        return max(2, min(3, len(df_customer_features)))

    # Sortiert nach höchstem Silhouette (beste Clusterqualität),
    # bei Gleichstand gewinnt kleineres k (einfachere Lösung bevorzugt).
    best_row = tbl.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]

    # Gibt bestes k zurück
    return int(best_row["k"])

def segment(df_customer_features: pd.DataFrame, k: int = 3) -> Tuple[np.ndarray, dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    # Selektiert numerische Features (wieder wie oben)
    feats_df = df_customer_features.select_dtypes(include=[float, int]).copy()
    X, scaler, cols = _scale_features(df_customer_features)

    # Fitte KMeans mit gegebener Clusteranzahl
    km = KMeans(n_clusters=int(k), n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Qualität des Clusterings: Silhouette Score
    sil = metrics.silhouette(X, labels)

    # Clustergrößen: wie viele Kunden pro Cluster?
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    # Für die Visualisierung: PCA auf 2 Dimensionen (für 2D-Plot)
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)

    # DataFrame für Plot: Koordinaten, Achsenlabels, Clusterlabels
    plot_df = pd.DataFrame({"x": XY[:, 0], "y": XY[:, 1]})
    plot_df["x_label"], plot_df["y_label"] = "PC1", "PC2"
    plot_df["cluster"] = labels

    # Clusterprofile: Mittelwert, Median, Count je Feature & Cluster
    prof = feats_df.copy()
    prof["cluster"] = labels
    profile_table = prof.groupby("cluster").agg(["mean", "median", "count"]).sort_index()

    # Rückgabe:
    # - labels: Clusterzuordnung für jeden Kunden
    # - dict: Metriken (Silhouette, Clustergrößen)
    # - cluster_centers_: Zentren im Feature-Space
    # - plot_df: Daten für 2D-Plot
    # - profile_table: aggregierte Profilwerte je Cluster
    return labels, {"silhouette": sil, "sizes": sizes}, km.cluster_centers_, plot_df, profile_table

