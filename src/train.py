"""
train.py

Trains and evaluates Logistic Regression, QDA and KNN on the cleaned bridge
earthquake dataset (data/bridge_earthquake.csv, produced by convert_data.py).

Pipeline
  1. Stratified 70/30 train/test split (test set is only used at the end).
  2. Each model is a Pipeline(StandardScaler -> model), so the scaler is fit
     on training folds only (no leakage).
  3. Hyperparameters are tuned with GridSearchCV (5-fold stratified CV,
     scored on F1) on the TRAINING set only.
       - KNN: k = 1..15
       - Logistic Regression: C and class_weight
       - QDA: reg_param
  4. Final evaluation on the held-out test set: accuracy, precision, recall,
     F1, ROC-AUC, confusion matrices, ROC curves.
  5. Oversampling experiment: the minority class of the TRAINING set is
     bootstrapped up to the majority size, the already-tuned models are
     refit, and evaluated on the same untouched test set.
  6. Fitted models and best parameters are saved so demo.py uses exactly the
     models that were evaluated.

Usage:
    python src/convert_data.py
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ------------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bridge_earthquake.csv"
OUT = ROOT / "outputs"
MODEL_DIR = OUT / "models"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["bridge_age", "magnitude", "distance_km"]  # convert_data.py output
TARGET = "damage"
SEED = 42
K_RANGE = list(range(1, 16))


def slug(name):
    """File-name-safe version of a model name."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace("+", "plus")
    )


def evaluate_model(name, model, X_test, y_test):
    """Score a fitted model on the test set; save confusion matrix + report."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    report = classification_report(
        y_test, pred, target_names=["No Damage", "Damage"], zero_division=0
    )
    (OUT / f"{slug(name)}_report.txt").write_text(report)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Damage", "Damage"], yticklabels=["No Damage", "Damage"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(OUT / f"cm_{slug(name)}.png", dpi=160)
    plt.close()

    print(f"{name:38s} acc={metrics['accuracy']:.3f}  prec={metrics['precision']:.3f}  "
          f"rec={metrics['recall']:.3f}  f1={metrics['f1']:.3f}  auc={metrics['roc_auc']:.3f}  "
          f"(TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]})")
    return metrics


def make_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def tune(name, pipeline, grid, cv, X_train, y_train):
    """GridSearchCV on the training set only, scored on F1."""
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"{name:20s} best params = {search.best_params_}  CV F1 = {search.best_score_:.3f}")
    return search


def main():
    # ------------------------------------------------------------------
    # Load and validate
    # ------------------------------------------------------------------
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\nRun `python src/convert_data.py` first."
        )
    df = pd.read_csv(DATA)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    df = df[FEATURES + [TARGET]].dropna()
    df[TARGET] = df[TARGET].astype(int)
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("The damage column must contain only 0 and 1.")

    print(f"Dataset: {df.shape[0]} rows | not damaged = {(df[TARGET] == 0).sum()}, "
          f"damaged = {(df[TARGET] == 1).sum()}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    print(f"Train = {len(X_train)} (damaged {int(y_train.sum())}) | "
          f"Test = {len(X_test)} (damaged {int(y_test.sum())})\n")

    # ------------------------------------------------------------------
    # Hyperparameter tuning (training set only)
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("Hyperparameter tuning (GridSearchCV, 5-fold stratified, scoring = F1)")

    lr_search = tune(
        "Logistic Regression",
        make_pipeline(LogisticRegression(max_iter=2000)),
        {"model__C": [0.01, 0.1, 1, 10, 100], "model__class_weight": [None, "balanced"]},
        cv, X_train, y_train,
    )
    qda_search = tune(
        "QDA",
        make_pipeline(QuadraticDiscriminantAnalysis()),
        {"model__reg_param": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]},
        cv, X_train, y_train,
    )
    knn_search = tune(
        "KNN",
        make_pipeline(KNeighborsClassifier()),
        {"model__n_neighbors": K_RANGE},
        cv, X_train, y_train,
    )
    best_k = knn_search.best_params_["model__n_neighbors"]

    # Save the full k = 1..15 tuning table and plot
    cvr = pd.DataFrame(knn_search.cv_results_)
    tuning = pd.DataFrame({
        "k": cvr["param_model__n_neighbors"].astype(int),
        "mean_f1": cvr["mean_test_score"],
        "std_f1": cvr["std_test_score"],
    })
    tuning.to_csv(OUT / "knn_hyperparameter_tuning.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.errorbar(tuning["k"], tuning["mean_f1"], yerr=tuning["std_f1"], marker="o", capsize=4)
    plt.axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"best k = {best_k}")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Mean F1 (5-fold stratified CV)")
    plt.title("KNN tuning with GridSearchCV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "knn_hyperparameter_tuning.png", dpi=160)
    plt.close()

    # ------------------------------------------------------------------
    # Final test-set evaluation (GridSearchCV refits the best model on all
    # of the training set, so best_estimator_ is ready to use)
    # ------------------------------------------------------------------
    knn_name = f"KNN (k={best_k})"
    fitted = {
        "Logistic Regression": lr_search.best_estimator_,
        "QDA": qda_search.best_estimator_,
        knn_name: knn_search.best_estimator_,
    }

    print("\nTEST-SET RESULTS (original training data)")
    results = [evaluate_model(n, m, X_test, y_test) for n, m in fitted.items()]
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "model_metrics.csv", index=False)

    results_df.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1)
    )
    plt.ylabel("Score")
    plt.title("Test-set model comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=160)
    plt.close()

    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, model in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc_value = results_df.loc[results_df["model"] == name, "roc_auc"].iloc[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=160)
    plt.close()

    # ------------------------------------------------------------------
    # Oversampling experiment (training set only)
    # ------------------------------------------------------------------
    train = X_train.copy()
    train[TARGET] = y_train
    majority = train[train[TARGET] == 0]
    minority = train[train[TARGET] == 1]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=SEED)
    balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=SEED)
    print(f"\nOversampled training set: {len(majority)} undamaged + {len(minority_up)} damaged "
          f"(bootstrap of {len(minority)} originals)")

    print("TEST-SET RESULTS (training-only oversampling, same test set)")
    over_results = []
    for name, model in fitted.items():
        refit = clone(model).fit(balanced[FEATURES], balanced[TARGET])
        over_results.append(evaluate_model(f"{name} + oversampling", refit, X_test, y_test))
    pd.DataFrame(over_results).to_csv(OUT / "model_metrics_with_oversampling.csv", index=False)

    # ------------------------------------------------------------------
    # Save models + best parameters for demo.py
    # ------------------------------------------------------------------
    for name, model in fitted.items():
        joblib.dump(model, MODEL_DIR / f"{slug(name)}.joblib")
    best_params = {
        "logistic_regression": lr_search.best_params_,
        "qda": qda_search.best_params_,
        "knn": knn_search.best_params_,
        "knn_best_k": int(best_k),
        "cv_f1": {
            "logistic_regression": lr_search.best_score_,
            "qda": qda_search.best_score_,
            "knn": knn_search.best_score_,
        },
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    (OUT / "best_params.json").write_text(json.dumps(best_params, indent=2))

    print(f"\nDone. Results, plots and models saved in: {OUT}")


if __name__ == "__main__":
    main()
