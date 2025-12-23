# src/02_add_deprivation.py
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR
from utils import ensure_dirs

def try_find_csv(must_contain):
    for p in RAW_DIR.rglob("*.csv"):
        name = p.name.lower()
        if all(s.lower() in name for s in must_contain):
            return p
    return None

def main():
    ensure_dirs(PROCESSED_DIR)

    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")

    onspd_path = try_find_csv(["onspd"])  # you don't have it in screenshot
    imd_path = try_find_csv(["file", "7"])  # you DO have this

    print("ONSPD:", onspd_path)
    print("IMD:", imd_path)

    merged = panel.copy()
    merged["LSOA11CD"] = pd.NA
    merged["imd_rank"] = pd.NA
    merged["imd_decile"] = pd.NA
    merged["imd_score"] = pd.NA

    if onspd_path is None or imd_path is None:
        print("⚠️ Missing ONSPD or IMD file. Saving panel_with_imd.parquet with empty IMD fields.")
    else:
        onspd = pd.read_csv(onspd_path, low_memory=False)

        # detect columns
        possible_pc = [c for c in onspd.columns if c.lower() in {"pcds", "postcode", "pcd"}]
        possible_lsoa = [c for c in onspd.columns if c.lower() in {"lsoa11", "lsoa11cd", "lsoa"}]
        if not possible_pc or not possible_lsoa:
            raise ValueError(f"ONSPD missing postcode/LSOA columns. Columns: {list(onspd.columns)[:50]}")

        pc_col = possible_pc[0]
        lsoa_col = possible_lsoa[0]
        onspd_small = onspd[[pc_col, lsoa_col]].rename(columns={pc_col: "Postcode", lsoa_col: "LSOA11CD"})

        imd = pd.read_csv(imd_path, low_memory=False)

        rename_map = {}
        for c in imd.columns:
            cl = c.lower()
            if "lsoa" in cl and "code" in cl:
                rename_map[c] = "LSOA11CD"
            if "index of multiple deprivation" in cl and "rank" in cl:
                rename_map[c] = "imd_rank"
            if "index of multiple deprivation" in cl and "decile" in cl:
                rename_map[c] = "imd_decile"
            if "index of multiple deprivation" in cl and "score" in cl:
                rename_map[c] = "imd_score"

        imd = imd.rename(columns=rename_map)
        needed = ["LSOA11CD", "imd_rank", "imd_decile", "imd_score"]
        missing = [c for c in needed if c not in imd.columns]
        if missing:
            raise ValueError(f"IMD file missing {missing}. Columns: {list(imd.columns)}")

        imd = imd[needed].dropna(subset=["LSOA11CD"])

        merged = (
            panel
            .merge(onspd_small, on="Postcode", how="left")
            .merge(imd, on="LSOA11CD", how="left")
        )

    out = PROCESSED_DIR / "panel_with_imd.parquet"
    merged.to_parquet(out, index=False)
    print("✅ Saved:", out)
    print("IMD coverage (imd_decile not null):", merged["imd_decile"].notna().mean())

if __name__ == "__main__":
    main()
