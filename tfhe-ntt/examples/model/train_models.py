#!/usr/bin/env python3
"""Train classifiers to predict the fastest algorithm label (r2/r4/rs).

Dataset:
  tfhe-ntt/examples/model/Dataset.csv

Features used:
  raw_N, padded_N,
  frac_2i_nonzero, frac_2i1_nonzero,
  frac_4i_nonzero, frac_4i1_nonzero, frac_4i2_nonzero, frac_4i3_nonzero

Target label:
  time_min_label  (values: r2, r4, rs)

Note:
- Rows with missing time_min_label (synthetic rows appended with blanks) are
  dropped for supervised training.
- We report both accuracy and balanced accuracy, plus confusion matrices.

Models:
- RandomForestClassifier(class_weight='balanced')
- GradientBoosting: try XGBoost if installed, otherwise HistGradientBoosting

Usage:
  python3 tfhe-ntt/examples/model/train_models.py
"""

from __future__ import annotations

import csv
import os
from collections import Counter

import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "raw_N",
    "padded_N",
    "frac_2i_nonzero",
    "frac_2i1_nonzero",
    "frac_4i_nonzero",
    "frac_4i1_nonzero",
    "frac_4i2_nonzero",
    "frac_4i3_nonzero",
]
TARGET = "time_min_label"


def load_xy(path: str):
    X = []
    y = []

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tgt = (row.get(TARGET) or "").strip()
            if tgt == "":
                continue
            try:
                feats = [float(row[c]) for c in FEATURES]
            except ValueError:
                # If any feature column is blank/non-numeric, skip the row
                continue
            X.append(feats)
            y.append(tgt)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    return X, y


def print_metrics(name: str, y_true, y_pred, labels_order):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    print(f"\n== {name} ==")
    print(f"accuracy: {acc:.6f}")
    print(f"balanced_accuracy: {bacc:.6f}")
    print("confusion_matrix (rows=true, cols=pred; label order: %s)" % (labels_order,))
    print(cm)


def try_train_xgboost(X_train, y_train_enc, X_test, y_test_enc, le: LabelEncoder):
    try:
        import xgboost as xgb  # type: ignore

        # Compute balanced class weights (like sklearn's class_weight='balanced')
        n_samples = len(y_train_enc)
        n_classes = len(le.classes_)
        counts = Counter(y_train_enc.tolist())
        # weight for class c = n_samples / (n_classes * count_c)
        class_weights = {c: n_samples / (n_classes * counts[c]) for c in counts}
        sample_weights = np.asarray([class_weights[c] for c in y_train_enc], dtype=np.float32)

        clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softmax",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=0,
            use_label_encoder=False,
        )

        clf.fit(X_train, y_train_enc, sample_weight=sample_weights)
        pred_enc = clf.predict(X_test)
        y_pred = le.inverse_transform(pred_enc)
        y_true = le.inverse_transform(y_test_enc)
        return ("XGBoost", y_true, y_pred)

    except ModuleNotFoundError:
        return None


def train_hist_gb(X_train, y_train_enc, X_test, y_test_enc, le: LabelEncoder):
    from sklearn.ensemble import HistGradientBoostingClassifier

    # HistGradientBoosting supports sample_weight
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=8,
        max_iter=500,
        random_state=0,
    )

    counts = Counter(y_train_enc.tolist())
    w = np.asarray([1.0 / counts[c] for c in y_train_enc], dtype=np.float32)

    clf.fit(X_train, y_train_enc, sample_weight=w)
    pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(pred_enc)
    y_true = le.inverse_transform(y_test_enc)
    return ("HistGradientBoosting", y_true, y_pred)


def main() -> int:
    dataset_path = os.path.join("tfhe-ntt", "examples", "model", "Dataset.csv")

    X, y = load_xy(dataset_path)
    print("Loaded rows with non-empty target:", X.shape[0])
    print("Feature dims:", X.shape[1])
    print("Target distribution:", dict(Counter(y.tolist())))

    # Encode labels to stable integer classes
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    labels_order = list(le.classes_)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=0,
        stratify=y_enc,
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=800,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    print_metrics(
        "RandomForest",
        le.inverse_transform(y_test),
        le.inverse_transform(pred_rf),
        labels_order,
    )

    # Gradient boosting model
    xgb_res = try_train_xgboost(X_train, y_train, X_test, y_test, le)
    if xgb_res is not None:
        name, y_true, y_pred = xgb_res
        print_metrics(name, y_true, y_pred, labels_order)
    else:
        name, y_true, y_pred = train_hist_gb(X_train, y_train, X_test, y_test, le)
        print_metrics(name, y_true, y_pred, labels_order)
        print("\nNote: xgboost is not installed; used scikit-learn HistGradientBoosting instead.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
