# ERP AI App — Forecast & Segmentation

## Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```
Beispiele unter `examples/`.

- **TensorFlow** ist optional (LSTM). Unter Windows/py311 ist `2.15.1` gepinnt.
- **Prophet** installiert als `prophet` (nicht `fbprophet`).

## Funktionen
- Absatzprognose (Prophet / LSTM, AutoML-Vergleich)
- IQR-Ausreißer-Handling (optional)
- Backtest-Metriken (RMSE/NRMSE)
- Kundensegmentierung (K-Means++), Auto-k via Silhouette
- CSV/PNG-Exporte
```
