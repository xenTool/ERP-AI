from __future__ import annotations
import pandas as pd
from pathlib import Path

# ==== INPUT ====
ITEM_ID  = "FOODS_3_586"         # <- Artikelnummer
STORE_ID = None                  # z.B. "WI_3" oder None für alle Stores
DATA_DIR = Path(".")             # Ordner mit sales_train_validation.csv & calendar.csv
# ===============

sales_path = DATA_DIR / "sales_train_validation.csv"
calendar_path = DATA_DIR / "calendar.csv"

if not sales_path.exists():
    raise FileNotFoundError(f"Nicht gefunden: {sales_path}")
if not calendar_path.exists():
    raise FileNotFoundError(f"Nicht gefunden: {calendar_path}")

# Original M5-Datensatz laden
df = pd.read_csv(sales_path)

# Auswahl des Artikels (und optional eines Stores)
mask = df["item_id"] == ITEM_ID
if STORE_ID:
    mask &= df["store_id"] == STORE_ID
item_df = df[mask]

if item_df.empty:
    raise ValueError(f"Keine Zeilen gefunden für ITEM_ID={ITEM_ID}" + (f" und STORE_ID={STORE_ID}" if STORE_ID else ""))

# Transformation ins Long Format
value_vars = [c for c in item_df.columns if c.startswith('d_')]
long_df = item_df.melt(
    id_vars=['id', 'item_id', 'store_id', 'state_id', 'dept_id', 'cat_id'],
    value_vars=value_vars,
    var_name='d',
    value_name='qty'
)

# Spaltenname vereinheitlichen und numerischen Tag extrahieren
long_df["d"] = long_df["d"].str.replace("d_", "", regex=False).astype(int)

# Kalenderdatei laden, um d_1 ... d_1913 in echte Datumsangaben umzuwandeln
calendar = pd.read_csv(calendar_path)[["d", "date"]]
calendar["d"] = calendar["d"].str.replace("d_", "", regex=False).astype(int)

# Join durchführen → echtes Datum zuweisen
merged = long_df.merge(calendar, on="d").drop(columns="d")

# Spaltenreihenfolge anpassen
merged = merged[["date", "item_id", "store_id", "state_id", "dept_id", "cat_id", "qty"]]

# CSV-Datei speichern (Name aus ITEM_ID + optional STORE_ID ableiten)
suffix = f"_{STORE_ID}" if STORE_ID else ""
out_name = f"{ITEM_ID}{suffix}_long.csv"
out_path = DATA_DIR / out_name
merged.to_csv(out_path, index=False)

print(f"Long-Format-Datei erfolgreich erstellt: {out_path}")
print(merged.head())
