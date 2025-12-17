#!/usr/bin/env python3
"""Train baseline ML classifiers (RandomForest + XGBoost) on final_dataset.csv.

Goal
----
Predict `output` ∈ {r2, r4, rs} from these input features:
  raw_N, padded_N,
  frac_2i_nonzero, frac_2i1_nonzero,
  frac_4i_nonzero, frac_4i1_nonzero, frac_4i2_nonzero, frac_4i3_nonzero

Splits
------
- train: 70%
- val:   15%
- test:  15%
Stratified by label.

Notes
-----
- We ignore the `polynomial` and multiplication-count columns.
- XGBoost is optional. If xgboost isn't installed, we fall back to
  HistGradientBoostingClassifier.

Outputs
-------
Prints accuracy on train/val/test for each model, plus a confusion matrix on test.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "raw_N",
    "padded_N",
    "frac_2i_nonzero",
    "frac_2i1_nonzero",
    "frac_4i_nonzero",
    "frac_4i1_nonzero",
    "frac_4i2_nonzero",
    "frac_4i3_nonzero",
]
LABEL_COL = "output"


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(str).copy()

    # Basic cleanup
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        # For safety: drop rows with NaNs in features
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]

    return X, y


def build_rf(random_state: int) -> Pipeline:
    # RF doesn't need scaling, but keeping a consistent pipeline is handy.
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURE_COLS),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_xgb_or_fallback(random_state: int) -> Tuple[str, Pipeline]:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), FEATURE_COLS),
        ],
        remainder="drop",
    )

    try:
        from xgboost import XGBClassifier  # type: ignore

        # We'll set objective/num_class after we know how many labels we have.
        clf = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=random_state,
        )
        name = "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=8,
            max_iter=500,
            random_state=random_state,
        )
        name = "hist_gbdt_fallback"

    return name, Pipeline([("pre", pre), ("clf", clf)])


def configure_xgb_for_classes(model: Pipeline, n_classes: int) -> None:
    """Configure XGBoost objective based on class count."""
    clf = model.named_steps.get("clf")
    # Only applies to XGBClassifier
    if clf is None or clf.__class__.__name__ != "XGBClassifier":
        return

    if n_classes <= 2:
        clf.set_params(objective="binary:logistic", eval_metric="logloss")
    else:
        clf.set_params(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)


def eval_model(model: Pipeline, Xtr, ytr, Xv, yv, Xte, yte) -> Dict[str, float]:
    model.fit(Xtr, ytr)
    p_tr = model.predict(Xtr)
    p_v = model.predict(Xv)
    p_te = model.predict(Xte)
    return {
        "train_acc": float(accuracy_score(ytr, p_tr)),
        "train_bal_acc": float(balanced_accuracy_score(ytr, p_tr)),
        "val_acc": float(accuracy_score(yv, p_v)),
        "val_bal_acc": float(balanced_accuracy_score(yv, p_v)),
        "test_acc": float(accuracy_score(yte, p_te)),
        "test_bal_acc": float(balanced_accuracy_score(yte, p_te)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "final_dataset.csv"),
        help="Path to final_dataset.csv",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y_str = load_dataset(args.csv)

    # Encode labels for estimators that require numeric classes (e.g., XGBoost)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_str), index=y_str.index)

    # train vs temp( val+test )
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=args.seed,
        stratify=y,
    )
    # val vs test
    Xv, Xte, yv, yte = train_test_split(
        Xtmp,
        ytmp,
        test_size=0.50,
        random_state=args.seed,
        stratify=ytmp,
    )

    print(f"Loaded rows: {len(X)}")
    print("Label distribution:")
    vc = y_str.value_counts()
    print(vc.to_string())

    models: List[Tuple[str, Pipeline]] = []
    models.append(("random_forest", build_rf(args.seed)))
    name2, m2 = build_xgb_or_fallback(args.seed)
    configure_xgb_for_classes(m2, n_classes=len(le.classes_))
    models.append((name2, m2))

    for name, model in models:
        metrics = eval_model(model, Xtr, ytr, Xv, yv, Xte, yte)
        print("\n==", name, "==")
        print("train_acc", f"{metrics['train_acc']:.4f}")
        print("train_bal_acc", f"{metrics['train_bal_acc']:.4f}")
        print("val_acc  ", f"{metrics['val_acc']:.4f}")
        print("val_bal_acc  ", f"{metrics['val_bal_acc']:.4f}")
        print("test_acc ", f"{metrics['test_acc']:.4f}")
        print("test_bal_acc ", f"{metrics['test_bal_acc']:.4f}")

        pred = model.predict(Xte)
        # Decode to original string labels for readability
        pred_str = le.inverse_transform(pred.astype(int))
        yte_str = le.inverse_transform(yte.astype(int))
        labels = list(le.classes_)
        cm = confusion_matrix(yte_str, pred_str, labels=labels)
        print("labels:", labels)
        print("confusion_matrix (rows=true, cols=pred):")
        print(cm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
