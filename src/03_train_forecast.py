# src/03_train_forecast.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from config import PROCESSED_DIR, OUT_TABLES
from utils import ensure_dirs

def make_features(df):
    df = df.sort_values(["Postcode", "Year"]).copy()
    df["lag1_total"] = df.groupby("Postcode")["total_kwh"].shift(1)
    df["lag2_total"] = df.groupby("Postcode")["total_kwh"].shift(2)
    df["yoy_total"] = df["total_kwh"] - df["lag1_total"]
    return df

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    ensure_dirs(OUT_TABLES)

    d = pd.read_parquet(PROCESSED_DIR / "panel_with_imd.parquet")
    d["imd_decile"] = d["imd_decile"].fillna(5)
    d = make_features(d).dropna(subset=["lag1_total", "total_kwh"])

    years = sorted(d["Year"].unique())
    rows = []

    for test_year in years[1:]:
        train = d[d["Year"] < test_year]
        test = d[d["Year"] == test_year]

        # Baseline
        base_pred = test["lag1_total"].values
        rows.append({
            "test_year": int(test_year),
            "model": "baseline_lag1",
            "MAE": mean_absolute_error(test["total_kwh"], base_pred),
            "RMSE": rmse(test["total_kwh"], base_pred),
        })

        feats = ["Year", "meters", "mean_kwh", "median_kwh", "lag1_total", "lag2_total", "yoy_total", "imd_decile"]
        train2 = train.dropna(subset=feats + ["total_kwh"])
        test2 = test.dropna(subset=feats + ["total_kwh"])

        Xtr, ytr = train2[feats], train2["total_kwh"]
        Xte, yte = test2[feats], test2["total_kwh"]

        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        rows.append({
            "test_year": int(test_year),
            "model": "HGBR",
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": rmse(yte, pred),
        })

    metrics = pd.DataFrame(rows)
    out = OUT_TABLES / "forecast_metrics.csv"
    metrics.to_csv(out, index=False)

    print("Saved:", out)
    print(metrics)

if __name__ == "__main__":
    main()