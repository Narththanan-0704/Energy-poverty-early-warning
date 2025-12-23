# src/utils.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pathlib import Path

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def standardise_uk_postcode(pc: str) -> str:
    if pd.isna(pc):
        return np.nan
    pc = str(pc).strip().upper()
    pc = re.sub(r"\s+", "", pc)
    if len(pc) >= 5:
        pc = pc[:-3] + " " + pc[-3:]
    return pc

def add_levels(df: pd.DataFrame, col="Postcode") -> pd.DataFrame:
    split = df[col].astype(str).str.split(" ", n=1, expand=True)
    df["OUTWARD"] = split[0]
    df["INWARD"] = split[1] if split.shape[1] > 1 else np.nan
    df["SECTOR"] = np.where(df["INWARD"].notna(), df["OUTWARD"] + " " + df["INWARD"].str[0], df["OUTWARD"])
    return df

def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    c = re.sub(r"[^a-z0-9() ]", "", c)
    return c

def detect_and_rename_electricity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises common headers into:
      Postcode, meters, total_kwh, mean_kwh, median_kwh
    Handles your sample format:
      POSTCODE, Number of meters, Consumption (kWh), Mean consumption (kWh), Median consumption (kWh)
    plus DESNZ variants.
    """
    cols = list(df.columns)
    norm = {_norm_col(c): c for c in cols}

    def find_col(fn):
        for n, orig in norm.items():
            if fn(n):
                return orig
        return None

    pc = find_col(lambda n: "postcode" in n)
    if pc is None:
        raise ValueError(f"Could not find postcode column. Columns: {cols}")

    meters = find_col(lambda n: ("meter" in n) and ("number" in n or "num" in n))
    if meters is None:
        meters = find_col(lambda n: n in {"meters", "num_meters"})

    mean = find_col(lambda n: ("mean" in n) and ("kwh" in n or "kw h" in n or "consumption" in n))
    median = find_col(lambda n: ("median" in n) and ("kwh" in n or "kw h" in n or "consumption" in n))

    # total consumption: must include kwh/consumption/total, but NOT mean/median
    total = find_col(lambda n: (("consumption" in n) or ("total" in n)) and ("mean" not in n) and ("median" not in n) and ("kwh" in n or "kw h" in n))
    if total is None:
        total = find_col(lambda n: n in {"total_conskwh", "total_kwh", "total cons kwh"})

    rename_map = {pc: "Postcode"}
    if meters: rename_map[meters] = "meters"
    if total: rename_map[total] = "total_kwh"
    if mean: rename_map[mean] = "mean_kwh"
    if median: rename_map[median] = "median_kwh"

    out = df.rename(columns=rename_map).copy()
    out["Postcode"] = out["Postcode"].apply(standardise_uk_postcode)

    for c in ["meters", "total_kwh", "mean_kwh", "median_kwh"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def find_first_csv(root: Path, must_contain: list[str]) -> Path:
    candidates = []
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        if all(s.lower() in name for s in must_contain):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {root} containing {must_contain}")
    return sorted(candidates)[0]
