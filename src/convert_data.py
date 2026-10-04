"""
convert_data.py

Converts the raw ds2.csv (from the Stanford CS229 bridge-earthquake
repository) into the clean CSV format expected by src/train.py:

    bridge_age, magnitude, distance_km, damage

Usage:
    python src/convert_data.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "ds2.csv"
OUT_PATH = ROOT / "data" / "bridge_earthquake.csv"

MILES_TO_KM = 1.60934

# Column names as they appear in the raw ds2.csv
RAW_COLUMNS = {
    "x2 = Age (yr)": "bridge_age",
    "x4 = Earthquake Magnitude": "magnitude",
    "x5 = Distance to epicenter (miles)": "distance_miles",
    "y1 = Damaged or not?": "damage",
}


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}\n"
            "Download ds2.csv from "
            "https://github.com/jzy95310/CS229-Fall-2018-Final-Report "
            "(Final Project Code/Data/ds2.csv) and place it at data/ds2.csv"
        )

    # The raw file uses latin1 encoding (has special characters)
    df = pd.read_csv(RAW_PATH, encoding="latin1")

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in ds2.csv: {missing}")

    # Select and rename the four columns we need
    df = df[list(RAW_COLUMNS.keys())].rename(columns=RAW_COLUMNS)

    # Drop rows with any missing values in these four columns
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after} incomplete rows ({before} -> {after})")

    # Convert distance from miles to km to match train.py's expected units
    df["distance_km"] = df["distance_miles"] * MILES_TO_KM
    df = df.drop(columns=["distance_miles"])

    # Damage must be an integer 0/1
    df["damage"] = df["damage"].astype(int)

    # Reorder columns to match train.py's FEATURES + TARGET
    df = df[["bridge_age", "magnitude", "distance_km", "damage"]]

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved cleaned dataset to {OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Class balance:\n{df['damage'].value_counts()}")
    print(f"\nPreview:\n{df.head()}")


if __name__ == "__main__":
    main()
