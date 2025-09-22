from __future__ import annotations
import pandas as pd
import re
from typing import Dict, Optional, Iterable, List, Tuple

# Pflichtspalten für valide Sales- bzw. Customer-Daten
REQUIRED_SALES = {"id", "date", "qty"}
REQUIRED_CUSTOMERS = {"customer_id", "order_id", "order_date", "net_amount"}

# Häufige Datumsformate (für robustes Parsing)
COMMON_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y %m %d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%d-%m-%Y",
)

def infer_schema(df: pd.DataFrame) -> dict:
    # Liefert einfaches Schema (Spalten, Datentypen, Zeilenanzahl).
    return {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "rows": len(df),
    }

def _parse_dates(series: pd.Series, formats: Iterable[str] = COMMON_DATE_FORMATS) -> pd.Series:
    # Versucht, eine Series mit bekannten Formaten zu parsen, sonst Fallback.
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            return parsed.dt.tz_localize(None)  # Zeitzone entfernen
        except Exception:
            pass
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.tz_localize(None)

def load_sales_df(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    # Normiert Sales-Daten (id, date, qty) nach Pflichtschema.
    out = pd.DataFrame()
    out["id"] = df[mapping["id"]].astype(str)
    out["date"] = _parse_dates(df[mapping["date"]])
    out["qty"] = pd.to_numeric(df[mapping["qty"]], errors="coerce")
    out = out.dropna(subset=["date", "qty"]).sort_values("date")
    missing = REQUIRED_SALES - set(out.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")
    return out

def load_customer_df(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    # Normiert Customer-Daten (IDs, Datum, Betrag, Rücksendung) nach Pflichtschema.
    out = pd.DataFrame()
    out["customer_id"] = df[mapping["customer_id"]].astype(str)
    out["order_id"] = df[mapping["order_id"]].astype(str)
    out["order_date"] = _parse_dates(df[mapping["order_date"]])
    out["net_amount"] = pd.to_numeric(df[mapping["net_amount"]], errors="coerce")

    # Optional: Rücksendespalte als bool
    if mapping.get("returned") and mapping["returned"] in df.columns:
        out["returned"] = df[mapping["returned"]].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        out["returned"] = False

    out = out.dropna(subset=["order_date", "net_amount"]).sort_values("order_date")
    missing = REQUIRED_CUSTOMERS - set(out.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")
    return out

#  Automatisches Spalten-Mapping
import re
from typing import List, Tuple

def _norm(s: str) -> str:
    # Normalisiert Spaltennamen: lower, alnum-only, ohne Leerzeichen/Unterstriche."""
    s = str(s).strip().lower()
    s = re.sub(r"[\s_\-]+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _best_match(columns: List[str], candidates: List[str]) -> str | None:
    # Wählt die beste Spalte aus 'columns', die einem der 'candidates' (normalisiert) am ähnlichsten ist.
    # Priorität: exakter Match (normalisiert) > startswith > enthält.

    cols_norm = {c: _norm(c) for c in columns}
    cand_norm = [_norm(c) for c in candidates]

    # 1) exakter Match
    for c, cn in cols_norm.items():
        if cn in cand_norm:
            return c
    # 2) beginnt mit
    for c, cn in cols_norm.items():
        if any(cn.startswith(k) for k in cand_norm):
            return c
    # 3) enthält
    for c, cn in cols_norm.items():
        if any(k in cn for k in cand_norm):
            return c
    return None

def suggest_mapping_sales(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    # Kandidatenlisten deutsch/englisch
    id_cands   = ["id", "productid", "sku", "artikel", "warengruppe", "gruppe", "itemid"]
    date_cands = ["date", "datum", "transdate", "orderdate", "bestelldatum"]
    qty_cands  = ["qty", "quantity", "menge", "absatz", "units", "salesqty", "anzahl"]

    return {
        "id":   _best_match(cols, id_cands),
        "date": _best_match(cols, date_cands),
        "qty":  _best_match(cols, qty_cands),
    }

def suggest_mapping_customers(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    cust_cands = ["customerid", "kunde", "kundennr", "kundekey", "customer"]
    orderid_cands = ["orderid", "bestellid", "auftragid", "ordernumber", "auftragsnr"]
    orderdate_cands = ["orderdate", "datum", "date", "bestelldatum", "transdate"]
    amount_cands = ["netamount", "amount", "umsatz", "netto", "total", "revenue"]
    returned_cands = ["returned", "retoure", "isreturn", "refund", "returnedflag"]

    return {
        "customer_id": _best_match(cols, cust_cands) or _best_match(cols, ["customer_id"]),
        "order_id":    _best_match(cols, orderid_cands) or _best_match(cols, ["order_id"]),
        "order_date":  _best_match(cols, orderdate_cands) or _best_match(cols, ["order_date"]),
        "net_amount":  _best_match(cols, amount_cands) or _best_match(cols, ["net_amount"]),
        "returned":    _best_match(cols, returned_cands) or (_best_match(cols, ["returned"]) or None),
    }
