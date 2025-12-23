# src/05_make_figures.py
import pandas as pd
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, OUT_FIG, OUT_TABLES
from utils import ensure_dirs

def main():
    ensure_dirs(OUT_FIG)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_with_imd.parquet")
    top = pd.read_csv(OUT_TABLES / "top50_risk_latest_year.csv")

    # 1) Trend
    yearly = panel.groupby("Year")["mean_kwh"].mean().reset_index()

    plt.figure()
    plt.plot(yearly["Year"], yearly["mean_kwh"], marker="o")
    plt.title("Average Mean Consumption (kWh) by Year")
    plt.xlabel("Year")
    plt.ylabel("Mean kWh")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "trend_mean_kwh.png")
    plt.close()

    # 2) Latest year boxplot by IMD decile
    latest_year = int(panel["Year"].max())
    latest = panel[panel["Year"] == latest_year].dropna(subset=["imd_decile", "mean_kwh"])

    deciles = sorted(latest["imd_decile"].dropna().unique())
    groups = [latest[latest["imd_decile"] == d]["mean_kwh"].dropna() for d in deciles]
    labels = [str(int(d)) for d in deciles]

    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(f"Mean kWh by IMD Decile (Latest Year = {latest_year})")
    plt.xlabel("IMD Decile (1 = most deprived)")
    plt.ylabel("Mean kWh")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "box_mean_kwh_by_imd_decile.png")
    plt.close()

    # 3) Top 20 risk postcodes
    top20 = top.sort_values("risk_score_0_100", ascending=False).head(20)

    plt.figure()
    plt.barh(top20["Postcode"], top20["risk_score_0_100"])
    plt.gca().invert_yaxis()
    plt.title(f"Top 20 Risk Postcodes (Year {latest_year})")
    plt.xlabel("Risk score (0–100)")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "top20_risk_postcodes.png")
    plt.close()

    print("Saved figures to:", OUT_FIG)

if __name__ == "__main__":
    main()