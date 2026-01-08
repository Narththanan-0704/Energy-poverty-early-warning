import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, OUT_TABLES, OUT_FIG
from utils import ensure_dirs

def safe_log10(x):
    # Avoid log of 0/negative
    x = pd.to_numeric(x, errors="coerce")
    return np.log10(np.where(x > 0, x, np.nan))

def main():
    ensure_dirs(OUT_TABLES, OUT_FIG)

    panel_path = PROCESSED_DIR / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing {panel_path}. Run: python src/01_build_panel.py first."
        )

    df = pd.read_parquet(panel_path)

    # -----------------------------
    # 1) Basic overview
    # -----------------------------
    overview = {
        "rows": [len(df)],
        "cols": [df.shape[1]],
        "min_year": [int(df["Year"].min())],
        "max_year": [int(df["Year"].max())],
        "unique_postcodes": [df["Postcode"].nunique()],
        "unique_outward": [df["OUTWARD"].nunique()] if "OUTWARD" in df.columns else [np.nan],
        "missing_meters_%": [round(df["meters"].isna().mean() * 100, 4)] if "meters" in df.columns else [np.nan],
        "missing_total_kwh_%": [round(df["total_kwh"].isna().mean() * 100, 4)] if "total_kwh" in df.columns else [np.nan],
        "missing_mean_kwh_%": [round(df["mean_kwh"].isna().mean() * 100, 4)] if "mean_kwh" in df.columns else [np.nan],
    }
    overview_df = pd.DataFrame(overview)
    overview_out = OUT_TABLES / "eda_overview.csv"
    overview_df.to_csv(overview_out, index=False)
    print("✅ Saved:", overview_out)

    # -----------------------------
    # 2) Yearly trends
    # -----------------------------
    yearly = df.groupby("Year").agg(
        total_kwh_sum=("total_kwh", "sum"),
        mean_kwh_avg=("mean_kwh", "mean"),
        meters_sum=("meters", "sum"),
        postcodes=("Postcode", "nunique"),
    ).reset_index()

    yearly_out = OUT_TABLES / "eda_yearly_summary.csv"
    yearly.to_csv(yearly_out, index=False)
    print("✅ Saved:", yearly_out)

    # Trend plot: average mean_kwh
    plt.figure()
    plt.plot(yearly["Year"], yearly["mean_kwh_avg"], marker="o")
    plt.title("EDA: Average Mean Consumption (kWh) by Year")
    plt.xlabel("Year")
    plt.ylabel("Average mean_kwh (kWh)")
    plt.tight_layout()
    fig1 = OUT_FIG / "eda_trend_mean_kwh.png"
    plt.savefig(fig1)
    plt.close()
    print("✅ Saved:", fig1)

    # Trend plot: total_kwh_sum (can be huge)
    plt.figure()
    plt.plot(yearly["Year"], yearly["total_kwh_sum"], marker="o")
    plt.title("EDA: Total Domestic Electricity Consumption (kWh) by Year")
    plt.xlabel("Year")
    plt.ylabel("Total_kwh (sum)")
    plt.tight_layout()
    fig2 = OUT_FIG / "eda_trend_total_kwh.png"
    plt.savefig(fig2)
    plt.close()
    print("✅ Saved:", fig2)

    # -----------------------------
    # 3) Distributions
    # -----------------------------
    # Histogram of mean_kwh
    if "mean_kwh" in df.columns:
        plt.figure()
        df["mean_kwh"].dropna().plot(kind="hist", bins=60)
        plt.title("EDA: Distribution of mean_kwh (all years)")
        plt.xlabel("mean_kwh")
        plt.ylabel("Count")
        plt.tight_layout()
        fig3 = OUT_FIG / "eda_hist_mean_kwh.png"
        plt.savefig(fig3)
        plt.close()
        print("✅ Saved:", fig3)

        # Log histogram (helps with skew)
        plt.figure()
        pd.Series(safe_log10(df["mean_kwh"])).dropna().plot(kind="hist", bins=60)
        plt.title("EDA: Distribution of log10(mean_kwh)")
        plt.xlabel("log10(mean_kwh)")
        plt.ylabel("Count")
        plt.tight_layout()
        fig4 = OUT_FIG / "eda_hist_log_mean_kwh.png"
        plt.savefig(fig4)
        plt.close()
        print("✅ Saved:", fig4)

    # Histogram of meters
    if "meters" in df.columns:
        plt.figure()
        df["meters"].dropna().plot(kind="hist", bins=60)
        plt.title("EDA: Distribution of meters (all years)")
        plt.xlabel("meters")
        plt.ylabel("Count")
        plt.tight_layout()
        fig5 = OUT_FIG / "eda_hist_meters.png"
        plt.savefig(fig5)
        plt.close()
        print("✅ Saved:", fig5)

    # -----------------------------
    # 4) Top 20 postcodes in latest year
    # -----------------------------
    latest_year = int(df["Year"].max())
    latest = df[df["Year"] == latest_year].copy()

    top20 = latest.sort_values("total_kwh", ascending=False).head(20)
    top20_out = OUT_TABLES / "eda_top20_total_kwh_latest_year.csv"
    top20.to_csv(top20_out, index=False)
    print("✅ Saved:", top20_out)

    # Bar chart
    plt.figure()
    plt.barh(top20["Postcode"], top20["total_kwh"])
    plt.gca().invert_yaxis()
    plt.title(f"EDA: Top 20 Postcodes by total_kwh (Year {latest_year})")
    plt.xlabel("total_kwh")
    plt.tight_layout()
    fig6 = OUT_FIG / "eda_top20_total_kwh_latest_year.png"
    plt.savefig(fig6)
    plt.close()
    print("✅ Saved:", fig6)

    # -----------------------------
    # 5) Sanity check: total_kwh vs meters*mean_kwh
    # -----------------------------
    if {"meters", "mean_kwh", "total_kwh"}.issubset(df.columns):
        tmp = df[["meters", "mean_kwh", "total_kwh"]].dropna().copy()
        tmp["meters_x_mean"] = tmp["meters"] * tmp["mean_kwh"]
        corr = tmp[["total_kwh", "meters_x_mean"]].corr().iloc[0, 1]

        sanity = pd.DataFrame({
            "metric": ["corr(total_kwh, meters*mean_kwh)"],
            "value": [corr]
        })
        sanity_out = OUT_TABLES / "eda_sanity_checks.csv"
        sanity.to_csv(sanity_out, index=False)
        print("✅ Saved:", sanity_out)

        # Scatter sample (to avoid plotting millions)
        sample = tmp.sample(n=min(20000, len(tmp)), random_state=42)
        plt.figure()
        plt.scatter(sample["meters_x_mean"], sample["total_kwh"], s=5)
        plt.title("EDA: total_kwh vs (meters * mean_kwh) [sample]")
        plt.xlabel("meters * mean_kwh")
        plt.ylabel("total_kwh")
        plt.tight_layout()
        fig7 = OUT_FIG / "eda_scatter_total_vs_metersxmean.png"
        plt.savefig(fig7)
        plt.close()
        print("✅ Saved:", fig7)

    # -----------------------------
    # 6) Optional: outward-level summary (simple)
    # -----------------------------
    if "OUTWARD" in df.columns:
        outward = df.groupby(["Year", "OUTWARD"]).agg(
            total_kwh_sum=("total_kwh", "sum"),
            mean_kwh_avg=("mean_kwh", "mean"),
            meters_sum=("meters", "sum"),
            postcodes=("Postcode", "nunique"),
        ).reset_index()

        outward_out = OUT_TABLES / "eda_outward_year_summary.csv"
        outward.to_csv(outward_out, index=False)
        print("✅ Saved:", outward_out)

    print("\n🎉 EDA complete. Check output/tables and output/figures")

if __name__ == "__main__":
    main()