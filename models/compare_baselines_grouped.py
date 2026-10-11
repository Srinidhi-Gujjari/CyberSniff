
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "paderborn_real_features.csv"
OUT = ROOT / "models" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["Healthy", "Physical_Fault"]
FAULT = "Physical_Fault"

FOUR_FEATURES = [
    "vibration_1_peak_to_peak",
    "vibration_1_min",
    "vibration_1_max",
    "vibration_1_kurtosis",
]

FOLDS = [
    ["K001", "KA03"],
    ["K002", "KA04"],
    ["K003", "KA05"],
]

df = pd.read_csv(DATA)

metadata = ["source_file", "bearing_id", "label", "damage_type"]
features = (
    df.drop(columns=metadata, errors="ignore")
    .select_dtypes(include="number")
    .columns.tolist()
)

if not set(FOUR_FEATURES).issubset(features):
    raise ValueError("The required four HGB features are missing.")

if set(df["label"].unique()) != set(LABELS):
    raise ValueError(f"Unexpected labels: {df['label'].unique()}")

# Use the same three grouped folds for every model.
for held_out in FOLDS:
    test = df[df["bearing_id"].isin(held_out)]
    train = df[~df["bearing_id"].isin(held_out)]
    if set(test["label"]) != set(LABELS):
        raise ValueError(f"Fold lacks both classes: {held_out}")
    if set(train["label"]) != set(LABELS):
        raise ValueError(f"Training fold lacks a class: {held_out}")

def scaled(model):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model,
    )

def unscaled(model):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        model,
    )

# All baseline models use the same full numeric feature set.
models = {
    "LogisticRegression": scaled(
        LogisticRegression(max_iter=2000, class_weight="balanced")
    ),
    "DecisionTree": unscaled(
        DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        )
    ),
    "RandomForest": unscaled(
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    ),
    "ExtraTrees": unscaled(
        ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    ),
    "SVM_RBF": scaled(
        SVC(
            kernel="rbf", class_weight="balanced",
            C=1.0, probability=False, random_state=42
        )
    ),
    "XGBoost": unscaled(
        XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    ),
    "HistGradientBoosting_full": make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            l2_regularization=1.0,
            random_state=42,
        ),
    ),
}

# Separate candidate: HGB with only the existing four selected features.
models["HistGradientBoosting_4_features"] = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        l2_regularization=1.0,
        random_state=42,
    ),
)

predictions = []
fold_metrics = []

for fold_no, held_out in enumerate(FOLDS, start=1):
    train_mask = ~df["bearing_id"].isin(held_out)
    test_mask = df["bearing_id"].isin(held_out)

    y_train = df.loc[train_mask, "label"]
    y_test = df.loc[test_mask, "label"]

    for name, template in models.items():
        selected = FOUR_FEATURES if name.endswith("_4_features") else features
        X_train = df.loc[train_mask, selected].apply(
            pd.to_numeric, errors="coerce"
        )
        X_test = df.loc[test_mask, selected].apply(
            pd.to_numeric, errors="coerce"
        )

        
        model = clone(template)

        if name == "XGBoost":
            y_train_encoded = y_train.map({
                "Healthy": 0,
                "Physical_Fault": 1,
            })
            model.fit(X_train, y_train_encoded)

            pred_encoded = model.predict(X_test)
            pred = pd.Series(pred_encoded, index=X_test.index).map({
                0: "Healthy",
                1: "Physical_Fault",
            }).to_numpy()
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)       
        cm = confusion_matrix(y_test, pred, labels=LABELS)
        tn, fp, fn, tp = cm.ravel()

        row = {
            "model": name,
            "fold": fold_no,
            "held_out_bearings": ",".join(held_out),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "fault_precision": precision_score(
                y_test, pred, pos_label=FAULT, zero_division=0
            ),
            "fault_recall": recall_score(
                y_test, pred, pos_label=FAULT, zero_division=0
            ),
            "fault_f1": f1_score(
                y_test, pred, pos_label=FAULT, zero_division=0
            ),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        }
        fold_metrics.append(row)

        fold_predictions = df.loc[
            test_mask, ["source_file", "bearing_id"]
        ].copy()
        fold_predictions["actual_label"] = y_test.to_numpy()
        fold_predictions["predicted_label"] = pred
        fold_predictions["model"] = name
        fold_predictions["fold"] = fold_no
        predictions.append(fold_predictions)

        print(
            f"{name:32s} fold={fold_no} "
            f"acc={row['accuracy']:.3f} "
            f"fault_recall={row['fault_recall']:.3f} "
            f"fault_f1={row['fault_f1']:.3f} "
            f"FPR={row['false_positive_rate']:.3f}"
        )

fold_df = pd.DataFrame(fold_metrics)
pred_df = pd.concat(predictions, ignore_index=True)

# Fold averages describe mean performance across the three held-out pairs.
metric_cols = [
    "accuracy", "balanced_accuracy", "fault_precision",
    "fault_recall", "fault_f1", "false_positive_rate",
]
summary = (
    fold_df.groupby("model")[metric_cols]
    .mean()
    .reset_index()
)

# Also calculate pooled out-of-fold metrics and combined confusion counts.
pooled = []
for name, group in pred_df.groupby("model"):
    cm = confusion_matrix(
        group["actual_label"], group["predicted_label"], labels=LABELS
    )
    tn, fp, fn, tp = cm.ravel()

    pooled.append({
        "model": name,
        "oof_accuracy": accuracy_score(
            group["actual_label"], group["predicted_label"]
        ),
        "oof_balanced_accuracy": balanced_accuracy_score(
            group["actual_label"], group["predicted_label"]
        ),
        "oof_fault_precision": precision_score(
            group["actual_label"], group["predicted_label"],
            pos_label=FAULT, zero_division=0
        ),
        "oof_fault_recall": recall_score(
            group["actual_label"], group["predicted_label"],
            pos_label=FAULT, zero_division=0
        ),
        "oof_fault_f1": f1_score(
            group["actual_label"], group["predicted_label"],
            pos_label=FAULT, zero_division=0
        ),
        "oof_false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    })

summary = summary.merge(pd.DataFrame(pooled), on="model")
summary = summary.sort_values(
    ["oof_fault_recall", "oof_fault_f1"], ascending=False
)

fold_df.to_csv(OUT / "baseline_grouped_fold_metrics.csv", index=False)
summary.to_csv(OUT / "baseline_grouped_comparison_summary.csv", index=False)
pred_df.to_csv(OUT / "baseline_grouped_oof_predictions.csv", index=False)

print("\nCOMPARISON SUMMARY")
print(summary.round(4).to_string(index=False))
print("\nConfusion matrices use rows=actual [Healthy, Physical_Fault]")
print("and columns=predicted [Healthy, Physical_Fault].")
print("\nSaved:")
print(OUT / "baseline_grouped_fold_metrics.csv")
print(OUT / "baseline_grouped_comparison_summary.csv")
print(OUT / "baseline_grouped_oof_predictions.csv")
print(
    "\nCAUTION: bearing ID is confounded with class. "
    "Results are experimental, not independent validation."
)