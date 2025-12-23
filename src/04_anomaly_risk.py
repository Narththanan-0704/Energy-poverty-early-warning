# src/04_anomaly_risk.py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from config import PROCESSED_DIR, OUT_TABLES
from utils import ensure_dirs

def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd != 0 else np.nan)

def main():
    ensure_dirs(OUT_TABLES)

    d = pd.read_parquet(PROCESSED_DIR / "panel_with_imd.parquet").copy()
    d["imd_decile"] = d["imd_decile"].fillna(5)
    d = d.sort_values(["Postcode", "Year"])

    d["lag1_total"] = d.groupby("Postcode")["total_kwh"].shift(1)
    d["yoy_total"] = d["total_kwh"] - d["lag1_total"]
    d = d.dropna(subset=["lag1_total", "total_kwh"])

    feats = ["Year", "meters", "mean_kwh", "median_kwh", "lag1_total", "yoy_total", "imd_decile"]
    d2 = d.dropna(subset=feats + ["total_kwh"]).copy()

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(d2[feats], d2["total_kwh"])

    d2["pred"] = model.predict(d2[feats])
    d2["residual"] = d2["total_kwh"] - d2["pred"]

    d2["resid_z"] = d2.groupby("OUTWARD")["residual"].transform(zscore)
    d2["anomaly_drop"] = d2["resid_z"] <= -2.5

    d2["imd_risk"] = 11 - d2["imd_decile"]
    d2["risk_raw"] = (0.6 * (-d2["resid_z"])) + (0.3 * (-zscore(d2["yoy_total"]))) + (0.1 * zscore(d2["imd_risk"]))
    d2["risk_score_0_100"] = 100 * d2["risk_raw"].rank(pct=True)

    latest_year = int(d2["Year"].max())
    latest = d2[d2["Year"] == latest_year].copy()

    top = latest.sort_values("risk_score_0_100", ascending=False).head(50)
    out1 = OUT_TABLES / "top50_risk_latest_year.csv"
    top[["Year","Postcode","OUTWARD","SECTOR","risk_score_0_100","resid_z","yoy_total","imd_decile"]].to_csv(out1, index=False)

    out2 = OUT_TABLES / "anomaly_drops_all_years.csv"
    d2[d2["anomaly_drop"]][["Year","Postcode","OUTWARD","resid_z","residual","pred","total_kwh","imd_decile"]].to_csv(out2, index=False)

    print("Saved:", out1)
    print("Saved:", out2)

if __name__ == "__main__":
    main()