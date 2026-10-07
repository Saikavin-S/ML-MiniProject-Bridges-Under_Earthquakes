"""
demo.py

Interactive demo: enter a bridge's age, an earthquake's magnitude and the
bridge's distance from the epicentre, and see what each model predicts.

The models loaded here are the SAME fitted models that train.py evaluated on
the test set (each one is a StandardScaler + classifier pipeline trained on
the 70% training split, with hyperparameters chosen by GridSearchCV).
Nothing is re-tuned or re-trained in this script.

Usage:
    python src/convert_data.py
    python src/train.py      # creates outputs/models/ and outputs/best_params.json
    python src/demo.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bridge_earthquake.csv"
MODEL_DIR = ROOT / "outputs" / "models"
PARAMS = ROOT / "outputs" / "best_params.json"
FEATURES = ["bridge_age", "magnitude", "distance_km"]


def load_models():
    if not PARAMS.exists() or not MODEL_DIR.exists():
        raise FileNotFoundError(
            "Trained models not found in outputs/. Run `python src/train.py` first."
        )
    best_k = json.loads(PARAMS.read_text())["knn_best_k"]
    models = {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.joblib"),
        "QDA": joblib.load(MODEL_DIR / "qda.joblib"),
        f"KNN (k={best_k})": joblib.load(MODEL_DIR / f"knn_k{best_k}.joblib"),
    }
    return models, best_k


def training_ranges():
    """Feature ranges in the dataset, used to warn about extrapolation."""
    df = pd.read_csv(DATA)
    return {c: (df[c].min(), df[c].max()) for c in FEATURES}


def ask(prompt, low=0.0):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if value < low:
            print(f"  Value must be at least {low}.")
            continue
        return value


def main():
    models, best_k = load_models()
    ranges = training_ranges() if DATA.exists() else {}

    print("=" * 64)
    print("Bridge Earthquake Damage Prediction - interactive demo")
    print("=" * 64)
    print(f"Models: Logistic Regression, QDA, KNN (k={best_k}), tuned with GridSearchCV.")
    print("Educational project only - not a real bridge safety tool.")
    print("Press Ctrl+C to quit.\n")

    while True:
        try:
            age = ask("Bridge age (years): ")
            magnitude = ask("Earthquake magnitude: ")
            distance = ask("Distance to epicentre (km): ")
        except (KeyboardInterrupt, EOFError):
            print("\nDemo ended.")
            sys.exit(0)

        row = pd.DataFrame([[age, magnitude, distance]], columns=FEATURES)

        print("\n" + "-" * 64)
        print(f"Input: age = {age:g} yr, magnitude = {magnitude:g}, distance = {distance:g} km")
        for column, value in zip(FEATURES, (age, magnitude, distance)):
            if column in ranges and not ranges[column][0] <= value <= ranges[column][1]:
                lo, hi = ranges[column]
                print(f"  Note: {column} is outside the dataset range ({lo:g} to {hi:g}); "
                      "predictions are extrapolation.")
        print("-" * 64)
        for name, model in models.items():
            label = "DAMAGED" if model.predict(row)[0] == 1 else "NOT DAMAGED"
            probability = model.predict_proba(row)[0][1]
            print(f"  {name:20s} -> {label:12s} P(damaged) = {probability:6.1%}")
        print("-" * 64 + "\n")


if __name__ == "__main__":
    main()
