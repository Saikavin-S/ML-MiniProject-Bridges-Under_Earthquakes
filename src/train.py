from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/bridge_earthquake.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
FEATURES = ["bridge_age", "magnitude", "distance_km"]
TARGET = "damage"

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}\nSee README.md for required columns.")
    df = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected: {required}")
    df = df[required].dropna()
    df[TARGET] = df[TARGET].astype(int)
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("damage must contain only 0 and 1.")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=None)),
        "QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    }
    rows=[]
    predictions={}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred=model.predict(X_test)
        predictions[name]=pred
        rows.append({"model":name,"accuracy":accuracy_score(y_test,pred),
                     "precision":precision_score(y_test,pred,zero_division=0),
                     "recall":recall_score(y_test,pred,zero_division=0),
                     "f1":f1_score(y_test,pred,zero_division=0)})
        (OUT/f"{name.lower().replace(' ','_').replace('(','').replace(')','')}_report.txt").write_text(
            classification_report(y_test,pred,zero_division=0))
        cm=confusion_matrix(y_test,pred,labels=[0,1])
        plt.figure(figsize=(4,3)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["No damage","Damage"],yticklabels=["No damage","Damage"])
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(name)
        plt.tight_layout(); plt.savefig(OUT/f"cm_{name.lower().replace(' ','_').replace('(','').replace(')','')}.png",dpi=160); plt.close()

    results=pd.DataFrame(rows)
    results.to_csv(OUT/"model_metrics.csv",index=False)
    results.set_index("model").plot(kind="bar",figsize=(9,5),ylim=(0,1))
    plt.ylabel("Score"); plt.title("Test-set model comparison"); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(OUT/"model_comparison.png",dpi=160); plt.close()

    # Oversampling is performed only on training data to avoid test leakage.
    train=pd.concat([X_train.reset_index(drop=True),y_train.reset_index(drop=True).rename(TARGET)],axis=1)
    counts=train[TARGET].value_counts()
    if len(counts)==2 and counts.min()!=counts.max():
        majority=train[train[TARGET]==counts.idxmax()]
        minority=train[train[TARGET]==counts.idxmin()]
        minority_up=resample(minority,replace=True,n_samples=len(majority),random_state=42)
        balanced=pd.concat([majority,minority_up]).sample(frac=1,random_state=42)
        for name, estimator in {
            "Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),
            "QDA":make_pipeline(StandardScaler(),QuadraticDiscriminantAnalysis(reg_param=0.1)),
            "KNN (k=5)":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5))
        }.items():
            estimator.fit(balanced[FEATURES],balanced[TARGET])
            pred=estimator.predict(X_test)
            rows.append({"model":name+" + oversampling","accuracy":accuracy_score(y_test,pred),
                         "precision":precision_score(y_test,pred,zero_division=0),
                         "recall":recall_score(y_test,pred,zero_division=0),
                         "f1":f1_score(y_test,pred,zero_division=0)})
        pd.DataFrame(rows).to_csv(OUT/"model_metrics_with_oversampling.csv",index=False)
    print(results.to_string(index=False))
    print(f"\nSaved metrics and plots to {OUT}")

if __name__=="__main__":
    main()
