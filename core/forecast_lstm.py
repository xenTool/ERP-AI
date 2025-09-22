
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from . import metrics

try:
    # TensorFlow und Keras importieren (nur wenn installiert)
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True      # Marker: TensorFlow ist verfügbar
    TF_ERROR = None
except Exception as e:
    # Falls Import fehlschlägt, Fehler merken → App kann ihn anzeigen
    TF_AVAILABLE = False
    TF_ERROR = repr(e)

def _to_sequences_next(y: np.ndarray, window: int):
    """
    Hilfsfunktion: Baut Sequenzen für LSTM.
    y = Zeitreihe (z. B. Absatz)
    window = Fenstergröße, wie viele Vergangenheitswerte als Input genutzt werden.
    """
    X, Y = [], []
    # Sliding-Window: Jeder Input X ist ein Block von "window"-Werten
    # und Y ist der nächste Wert, den das Modell lernen soll vorherzusagen.
    for i in range(len(y) - window):
        X.append(y[i:i+window])
        Y.append(y[i+window])
    return np.array(X), np.array(Y)


def fit_predict(ts_df: pd.DataFrame, horizon: int = 30, window: int = 30, epochs: int = 30):
    # Sicherheitscheck: Ist TensorFlow installiert? Sonst abbrechen.
    if not TF_AVAILABLE:
        raise ImportError(f"TensorFlow konnte nicht importiert werden: {TF_ERROR}")

    # Daten chronologisch sortieren (wichtig für Zeitreihen!)
    df = ts_df.sort_values("date")

    # Zielwerte (Absatzmenge) als NumPy-Array mit 1 Spalte
    y = df["qty"].astype(float).values.reshape(-1, 1)

    # Normalisierung in [0,1] mit MinMaxScaler
    # → sorgt für stabileres Training des LSTM
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y).flatten()

    # Prüfen, ob genug Datenpunkte vorhanden sind
    # Wir brauchen mindestens (Fenster + Horizont + 1)
    if len(y_scaled) < window + horizon + 1:
        raise ValueError("Zeitreihe zu kurz für LSTM mit gegebenem Fenster/Horizont.")

    # --- Holdout-Split für Validierung ---
    # Trainingsdaten = alles bis auf den letzten "horizon"-Block
    train_end = len(y_scaled) - int(horizon)
    y_train = y_scaled[:train_end]

    # Trainingssequenzen erzeugen:
    # Xtr = Input-Fenster, Ytr = nächster Wert
    Xtr, Ytr = _to_sequences_next(y_train, int(window))

    # Modellaufbau:
    # - LSTM Layer mit 64 Neuronen (merkt sich Muster in Sequenzen)
    # - Dense Layer mit 1 Neuron (gibt die Prognose aus)
    model = Sequential([
        LSTM(64, input_shape=(int(window), 1)),
        Dense(1)
    ])

    # Optimizer = Adam, Loss = MSE → Standard für Regressions-Zeitreihen
    model.compile(optimizer="adam", loss="mse")

    # EarlyStopping Callback:
    # bricht ab, wenn 5 Epochen keine Verbesserung mehr
    cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

    # Training:
    # - Input muss 3D sein: (Samples, Window, Features=1)
    # - 20% Validierung
    # - Batch-Größe 32
    model.fit(
        Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1)),
        Ytr,
        epochs=int(epochs),
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=cb
    )

    # --- Backtest (auf Holdout-Teil) ---
    # Start = letztes Fenster vor Holdout
    history = list(y_scaled[train_end-int(window):train_end])
    preds_hold = []

    # Horizon Schritte iterativ vorhersagen
    for _ in range(int(horizon)):
        x = np.array(history[-int(window):]).reshape((1, int(window), 1))
        p = model.predict(x, verbose=0)[0, 0]
        preds_hold.append(p)
        history.append(p)

    # Echte Holdout-Werte
    y_true_hold = y_scaled[train_end:]

    # Rücktransformation (Skalierung zurück auf Originalwerte)
    y_pred_hold = scaler.inverse_transform(np.array(preds_hold).reshape(-1,1)).flatten()
    y_true_hold_inv = scaler.inverse_transform(y_true_hold.reshape(-1,1)).flatten()

    # Fehlermaße berechnen
    rm = metrics.rmse(y_true_hold_inv, y_pred_hold)
    nrm = metrics.nrmse(y_true_hold_inv, y_pred_hold)

    # --- Zukunftsforecast ---
    # Start = letztes Fenster aus den Originaldaten
    history2 = list(y_scaled[-int(window):])
    preds_future = []

    # Horizon Schritte in die Zukunft simulieren
    for _ in range(int(horizon)):
        x = np.array(history2[-int(window):]).reshape((1, int(window), 1))
        p = model.predict(x, verbose=0)[0, 0]
        preds_future.append(p)
        history2.append(p)

    # Rückskalieren auf Originalwerte
    preds = scaler.inverse_transform(np.array(preds_future).reshape(-1, 1)).flatten()

    # Forecast-Daten mit Datum anreichern (ab letztem Datum + 1 Tag)
    dates = pd.date_range(
        start=df["date"].max() + pd.Timedelta(days=1),
        periods=int(horizon),
        freq="D"
    )
    out = pd.DataFrame({"date": dates, "yhat": preds})

    # Rückgabe: DataFrame mit Prognose & Metriken (RMSE, NRMSE)
    return out, {"rmse": rm, "nrmse": nrm}


def _to_sequences_next(y: np.ndarray, window: int):
    """
    Baut Trainingssequenzen für LSTM:
    - Input X = 'window' viele vergangene Werte
    - Ziel Y = nächster Wert nach dem Fenster
    """
    X, Y = [], []
    # Sliding Window über die Zeitreihe
    for i in range(len(y) - window):
        # Nimmt einen Block der Länge "window"
        X.append(y[i:i+window])
        # Zielwert = der Wert direkt nach diesem Fenster
        Y.append(y[i+window])
    # Rückgabe: NumPy-Arrays für Training
    return np.array(X), np.array(Y)


def fit_predict(ts_df: pd.DataFrame, horizon: int = 30, window: int = 30, epochs: int = 30):
    """
    Trainiert ein LSTM auf einer Zeitreihe und liefert:
    - Backtest-Metriken (RMSE, NRMSE)
    - Zukunftsprognose (horizon Tage)
    """
    # Sicherheitscheck: Ist TensorFlow verfügbar?
    if not TF_AVAILABLE:
        raise ImportError(f"TensorFlow konnte nicht importiert werden: {TF_ERROR}")

    # Zeitreihe chronologisch sortieren
    df = ts_df.sort_values("date")

    # Zielspalte "qty" in NumPy-Array umwandeln
    y = df["qty"].astype(float).values.reshape(-1, 1)

    # Skalieren auf [0,1] für stabileres Training
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y).flatten()

    # Datenlänge prüfen → muss mindestens (window + horizon + 1) sein
    if len(y_scaled) < window + horizon + 1:
        raise ValueError("Zeitreihe zu kurz für LSTM mit gegebenem Fenster/Horizont.")

    # --- Holdout-Split ---
    # Trainingsdaten = bis auf die letzten "horizon"-Werte
    train_end = len(y_scaled) - int(horizon)
    y_train = y_scaled[:train_end]

    # Sequenzen für Training erzeugen
    Xtr, Ytr = _to_sequences_next(y_train, int(window))

    # --- Modellarchitektur ---
    model = Sequential([
        # LSTM-Layer mit 64 Neuronen, Inputgröße = (Fenster, 1 Feature)
        LSTM(64, input_shape=(int(window), 1)),
        # Dense-Layer → gibt 1 Wert aus (nächster Absatz)
        Dense(1)
    ])
    # Optimizer Adam + Loss MSE = Standard für Zeitreihen-Regression
    model.compile(optimizer="adam", loss="mse")

    # EarlyStopping Callback:
    # Stoppt Training, wenn 5 Epochen keine Verbesserung mehr
    cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

    # Training starten
    model.fit(
        Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1)),  # 3D-Form (Samples, Window, Features=1)
        Ytr,
        epochs=int(epochs),
        batch_size=32,
        validation_split=0.2,  # 20% Validation
        verbose=0,
        callbacks=cb
    )

    # --- Backtest: Validierung auf Holdout ---
    # Start = letztes Fenster vor Holdout-Bereich
    history = list(y_scaled[train_end-int(window):train_end])
    preds_hold = []
    for _ in range(int(horizon)):
        # Nutzt immer die letzten "window"-Werte
        x = np.array(history[-int(window):]).reshape((1, int(window), 1))
        # Vorhersage für nächsten Schritt
        p = model.predict(x, verbose=0)[0, 0]
        preds_hold.append(p)
        history.append(p)  # Vorhersage in History aufnehmen → rollierendes Forecasting

    # Echte Werte im Holdout
    y_true_hold = y_scaled[train_end:]

    # Rückskalierung (Predictions & True Values)
    y_pred_hold = scaler.inverse_transform(np.array(preds_hold).reshape(-1,1)).flatten()
    y_true_hold_inv = scaler.inverse_transform(y_true_hold.reshape(-1,1)).flatten()

    # Fehlermaße berechnen
    rm = metrics.rmse(y_true_hold_inv, y_pred_hold)
    nrm = metrics.nrmse(y_true_hold_inv, y_pred_hold)

    # --- Zukunftsforecast ---
    # Start: letztes Fenster der Serie
    history2 = list(y_scaled[-int(window):])
    preds_future = []
    for _ in range(int(horizon)):
        x = np.array(history2[-int(window):]).reshape((1, int(window), 1))
        p = model.predict(x, verbose=0)[0, 0]
        preds_future.append(p)
        history2.append(p)

    # Rückskalieren auf Originalwerte
    preds = scaler.inverse_transform(np.array(preds_future).reshape(-1, 1)).flatten()

    # Forecast-Datenframe mit Datum anreichern
    dates = pd.date_range(
        start=df["date"].max() + pd.Timedelta(days=1),
        periods=int(horizon),
        freq="D"
    )
    out = pd.DataFrame({"date": dates, "yhat": preds})

    # Rückgabe: Forecast + Metriken
    return out, {"rmse": rm, "nrmse": nrm}
