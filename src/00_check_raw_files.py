# src/00_check_raw_files.py
from pathlib import Path
from config import RAW_DIR, YEARS
from utils import find_first_csv

def main():
    print("Checking raw data folder:", RAW_DIR.resolve())

    # Try to locate each year's electricity file
    for y in YEARS:
        f = find_first_csv(RAW_DIR, must_contain=[str(y), "electricity"])
        print(f"OK: Found electricity {y} -> {f}")

    # IMD file 7
    imd = find_first_csv(RAW_DIR, must_contain=["file", "7"])
    print("OK: Found IMD file candidate ->", imd)

    # ONSPD
    onspd = find_first_csv(RAW_DIR, must_contain=["onspd"])
    print("OK: Found ONSPD candidate ->", onspd)

    print("\nAll checks passed ✅")

if __name__ == "__main__":
    main()