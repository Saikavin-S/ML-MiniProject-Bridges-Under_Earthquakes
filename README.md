# Bridge Earthquake Damage Prediction
 
**Team:** Saikavin S [PES1UG24CS402], Sarayu Mayakonda [PES1UG24CS926]  
**Course:** UE24CS352A Machine Learning  
**Project:** Problem 27 - Mini Project  
**Duration:** September 29 - October 10, 2026
 
---
 
## Problem
Given a bridge's **age**, an earthquake's **magnitude**, and the bridge's **distance from the epicentre**, predict whether the bridge was **damaged (1)** or **not damaged (0)**. This is binary classification, following Jiang & Zhang, *Predicting Bridge Performance under Earthquakes with Supervised Learning* (Stanford CS229, 2018).

## Dataset
- Source: `ds2.csv` from the authors' repository (https://github.com/jzy95310/CS229-Fall-2018-Final-Report, `Final Project Code/Data/ds2.csv`).
- Raw file: 999 rows. After dropping rows with a missing age, magnitude, distance or damage label: **249 rows** (228 not damaged, 21 damaged, about 11:1).
- Features: `bridge_age` (years), `magnitude`, `distance_km` (converted from miles, x 1.60934). Target: `damage`.
- Only three of the dataset's columns are used; bridge material and structure type are not.

## Method
1. `convert_data.py` cleans `ds2.csv` into `data/bridge_earthquake.csv`.
2. Stratified 70/30 split (`random_state=42`): 174 train rows (15 damaged), 75 test rows (6 damaged).
3. Every model is a `Pipeline(StandardScaler -> classifier)`, so scaling is fit on training folds only.
4. **GridSearchCV** (5-fold stratified CV, scoring = F1, training set only):
   - KNN: `k = 1..15`
   - Logistic Regression: `C` in {0.01, 0.1, 1, 10, 100} and `class_weight` in {None, balanced}
   - QDA: `reg_param` in {0, 0.1, 0.25, 0.5, 0.75, 1.0}
5. Evaluate once on the held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves.
6. Experiment: bootstrap-oversample the damaged class **in the training set only**, refit the tuned models, and evaluate on the same untouched test set.

## Structure
```
data/ds2.csv                  raw data (999 rows)
data/bridge_earthquake.csv    cleaned data (generated)
src/convert_data.py           raw -> clean
src/train.py                  tuning, training, evaluation, plots, saves models
src/demo.py                   interactive predictions (uses the saved models)
outputs/                      metrics, plots, reports, saved models (generated)
```

## Run
```bash
pip install -r requirements.txt
python src/convert_data.py
python src/train.py
python src/demo.py
```
`demo.py` loads the exact models that `train.py` evaluated (from `outputs/models/`), so run `train.py` first.

## Results (test set: 75 bridges, 6 damaged)

Selected hyperparameters: KNN `k=1` (CV F1 0.841), Logistic Regression `C=1, class_weight=None` (CV F1 0.971), QDA `reg_param=0.25` (CV F1 1.000).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP / FN / FP |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.960 | 1.000 | 0.500 | 0.667 | 0.957 | 3 / 3 / 0 |
| **QDA** | **0.973** | **1.000** | **0.667** | **0.800** | **0.976** | 4 / 2 / 0 |
| KNN (k=1) | 0.920 | 0.500 | 0.333 | 0.400 | 0.652 | 2 / 4 / 2 |

With training-only oversampling (same test set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.947 | 0.667 | 0.667 | 0.667 | 0.952 |
| QDA | 0.893 | 0.417 | 0.833 | 0.556 | 0.976 |
| KNN (k=1) | 0.933 | 0.600 | 0.500 | 0.545 | 0.736 |

KNN tuning (mean F1 over 5 folds): k=1 0.841 (std 0.086), k=3, 4, 5 each 0.823 (std 0.028), k=2 0.703, then lower for larger k down to 0.200 at k=14 and 15. Full table: `outputs/knn_hyperparameter_tuning.csv`.

## Findings
- QDA is best on this split (F1 0.800, precision 1.000, recall 0.667, AUC 0.976).
- Accuracy is misleading: always predicting "not damaged" scores 0.920 on this test set while finding no damaged bridge.
- Every damaged bridge in the data has magnitude >= 6.6 (21 of the 35 bridges with magnitude >= 6.6 were damaged). Magnitude carries most of the signal, which is why the linear and Gaussian models do well and KNN with k=1 does poorly.
- Oversampling raises recall (QDA 0.667 -> 0.833, Logistic Regression 0.500 -> 0.667) but lowers precision, and QDA's F1 falls from 0.800 to 0.556. KNN improves from F1 0.400 to 0.545.

## Limitations
- 249 usable rows, 21 damaged; the test set has 6 damaged bridges, so one prediction moves recall by about 17 points. Differences between models on one split are not statistically reliable.
- CV F1 values (QDA 1.000) are optimistic: they are the maximum over a grid, computed from 15 damaged training examples. The test F1 is the more honest figure.
- "No recorded damage" is treated as undamaged, which may not always be true.
- Three features only. Not a real bridge-safety tool.

## Reference
Jiang, Z. & Zhang, X. (2018). *Predicting Bridge Performance under Earthquakes with Supervised Learning.* CS229 Final Report, Stanford University.
