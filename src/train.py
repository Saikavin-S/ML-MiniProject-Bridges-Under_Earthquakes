from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.utils import resample


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "bridge_earthquake.csv"

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# ============================================================
# 2. DATASET COLUMNS
# ============================================================

FEATURES = [
    "bridge_age",
    "magnitude",
    "distance_miles"
]

TARGET = "damage"


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Damage", "Damage"],
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Damage", "Damage"],
        yticklabels=["No Damage", "Damage"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(name)

    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace("+", "plus")
    )

    plt.savefig(
        OUT / f"cm_{filename}.png",
        dpi=160
    )

    plt.close()

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA}"
        )

    df = pd.read_csv(DATA)

    required_columns = FEATURES + [TARGET]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Expected columns: {required_columns}"
        )

    df = df[required_columns].dropna()

    df[TARGET] = df[TARGET].astype(int)

    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError(
            "The damage column must contain only 0 and 1."
        )

    print("\nDataset shape:")
    print(df.shape)

    print("\nClass distribution:")
    print(df[TARGET].value_counts())

    # --------------------------------------------------------
    # Split features and target
    # --------------------------------------------------------

    X = df[FEATURES]
    y = df[TARGET]

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # ========================================================
    # 5. LOGISTIC REGRESSION
    # ========================================================

    logistic_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ])

    logistic_params = {
        "model__C": [
            0.01,
            0.1,
            1,
            10,
            100
        ],

        "model__class_weight": [
            None,
            "balanced"
        ]
    }

    logistic_grid = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    logistic_grid.fit(
        X_train,
        y_train
    )

    print("\n\nLOGISTIC REGRESSION")
    print("-" * 60)

    print(
        "Best parameters:",
        logistic_grid.best_params_
    )

    print(
        "Best CV F1-score:",
        round(logistic_grid.best_score_, 4)
    )

    best_logistic = logistic_grid.best_estimator_

    # ========================================================
    # 6. QDA
    # ========================================================

    qda_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            QuadraticDiscriminantAnalysis()
        )
    ])

    qda_params = {
        "model__reg_param": [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0
        ]
    }

    qda_grid = GridSearchCV(
        estimator=qda_pipeline,
        param_grid=qda_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    qda_grid.fit(
        X_train,
        y_train
    )

    print("\n\nQDA")
    print("-" * 60)

    print(
        "Best parameters:",
        qda_grid.best_params_
    )

    print(
        "Best CV F1-score:",
        round(qda_grid.best_score_, 4)
    )

    best_qda = qda_grid.best_estimator_

    # ========================================================
    # 7. KNN
    # ========================================================

    knn_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier()
        )
    ])

    knn_params = {
        "model__n_neighbors": [
            3,
            5,
            7,
            9,
            11,
            15,
            21
        ],

        "model__weights": [
            "uniform",
            "distance"
        ],

        "model__p": [
            1,
            2
        ]
    }

    knn_grid = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=knn_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    knn_grid.fit(
        X_train,
        y_train
    )

    print("\n\nKNN")
    print("-" * 60)

    print(
        "Best parameters:",
        knn_grid.best_params_
    )

    print(
        "Best CV F1-score:",
        round(knn_grid.best_score_, 4)
    )

    best_knn = knn_grid.best_estimator_

    # ========================================================
    # 8. FINAL TEST-SET EVALUATION
    # ========================================================

    print("\n\n")
    print("=" * 60)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 60)

    results = []

    results.append(
        evaluate_model(
            "Logistic Regression",
            best_logistic,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "QDA",
            best_qda,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "KNN",
            best_knn,
            X_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUT / "model_metrics.csv",
        index=False
    )

    # ========================================================
    # 9. MODEL COMPARISON GRAPH
    # ========================================================

    results_plot = results_df.set_index("model")[
        [
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    ]

    results_plot.plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1)
    )

    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Model Performance Comparison")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUT / "model_comparison.png",
        dpi=160
    )

    plt.close()

    # ========================================================
    # 10. PRINT FINAL RESULTS
    # ========================================================

    print("\n\nFINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n")
    print("All results saved to:")
    print(OUT)

    print("\nGenerated files:")

    for file in sorted(OUT.iterdir()):
        print(" -", file.name)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
