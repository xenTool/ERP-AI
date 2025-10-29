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

## Lizenz

MIT License

Copyright (c) 2025 Peter Beiner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.