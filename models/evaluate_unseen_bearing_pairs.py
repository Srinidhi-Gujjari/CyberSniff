
import os
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)
OUT_DIR = os.path.join(ROOT, "models", "outputs", "bearing_pair_tests")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

metadata = ["source_file", "bearing_id", "label", "damage_type"]
features = df.drop(columns=metadata, errors="ignore") \
             .select_dtypes(include="number").columns.tolist()

healthy_ids = ["K001", "K002", "K003"]
fault_ids = ["KA03", "KA04", "KA05"]

model_factories = {
    "ExtraTrees": lambda: ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "HistGradientBoosting": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            l2_regularization=1.0,
            random_state=42
        )
    )
}

records = []

for healthy_id in healthy_ids:
    for fault_id in fault_ids:
        held_out = [healthy_id, fault_id]

        train = df[~df["bearing_id"].isin(held_out)].copy()
        test = df[df["bearing_id"].isin(held_out)].copy()

        print("\n" + "=" * 65)
        print(f"Held-out healthy: {healthy_id}; faulty: {fault_id}")
        print("Test counts:")
        print(test.groupby(["bearing_id", "label"]).size().to_string())

        for name, factory in model_factories.items():
            model = factory()

            # Fit imputation using training data only.
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(train[features])
            X_test = imputer.transform(test[features])

            model.fit(X_train, train["label"])
            pred = model.predict(X_test)

            fault_recall = recall_score(
                test["label"],
                pred,
                pos_label="Physical_Fault",
                zero_division=0
            )

            row = {
                "model": name,
                "held_out_healthy": healthy_id,
                "held_out_fault": fault_id,
                "accuracy": accuracy_score(test["label"], pred),
                "balanced_accuracy": balanced_accuracy_score(
                    test["label"], pred
                ),
                "fault_recall": fault_recall
            }
            records.append(row)

            print(
                f"{name}: accuracy={row['accuracy']:.3f}, "
                f"balanced_accuracy={row['balanced_accuracy']:.3f}, "
                f"fault_recall={fault_recall:.3f}"
            )

            print(classification_report(
                test["label"],
                pred,
                labels=["Healthy", "Physical_Fault"],
                zero_division=0
            ))

results = pd.DataFrame(records)
results.to_csv(
    os.path.join(OUT_DIR, "pair_results.csv"),
    index=False
)

print("\nMEAN RESULTS ACROSS NINE PAIRS")
print(
    results.groupby("model")[
        ["accuracy", "balanced_accuracy", "fault_recall"]
    ].mean().round(3).to_string()
)

print(f"\nSaved results to: {OUT_DIR}")
