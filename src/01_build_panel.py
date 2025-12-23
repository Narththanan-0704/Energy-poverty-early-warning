print("✅ RUNNING:", __file__)

# src/01_build_panel.py
import traceback
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, YEARS
from utils import ensure_dirs, add_levels, detect_and_rename_electricity_columns, find_first_csv

def load_one(year: int) -> pd.DataFrame:
    path = find_first_csv(RAW_DIR, must_contain=[str(year), "electricity"])
    print(f"\n=== YEAR {year} ===")
    print("File:", path)

    df = pd.read_csv(path, low_memory=False)
    print("Raw columns:", list(df.columns)[:20])

    df = detect_and_rename_electricity_columns(df)
    print("Renamed columns:", list(df.columns))

    df = df[df["Postcode"].str.lower() != "all postcodes"].copy()
    df["Year"] = year
    df = add_levels(df, col="Postcode")

    keep = ["Year", "Postcode", "OUTWARD", "SECTOR", "meters", "total_kwh", "mean_kwh", "median_kwh"]
    keep = [c for c in keep if c in df.columns]
    print("Keeping:", keep)
    print("Rows:", len(df))

    return df[keep]

def main():
    ensure_dirs(PROCESSED_DIR)

    frames = []
    for y in YEARS:
        frames.append(load_one(y))

    panel = pd.concat(frames, ignore_index=True).sort_values(["Postcode", "Year"])
    print("\nTOTAL PANEL ROWS:", len(panel))

    # Always save CSV (guaranteed)
    out_csv = PROCESSED_DIR / "panel.csv"
    panel.to_csv(out_csv, index=False)
    print("✅ Saved CSV:", out_csv)

    # Try parquet too (if pyarrow works)
    try:
        out_pq = PROCESSED_DIR / "panel.parquet"
        panel.to_parquet(out_pq, index=False)
        print("✅ Saved PARQUET:", out_pq)
    except Exception as e:
        print("⚠️ Parquet save failed (install pyarrow). Error:")
        print(e)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n❌ SCRIPT FAILED. Full error below:\n")
        traceback.print_exc()
