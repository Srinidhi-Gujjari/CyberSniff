
import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)

OUTPUT_DIR = os.path.join(ROOT, "data", "processed")
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "paderborn_cyber_simulation.csv"
)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

# Only use healthy measurements as the starting point
healthy = df[df["label"] == "Healthy"].copy().reset_index(drop=True)

if len(healthy) < 4:
    raise ValueError("Not enough healthy samples for simulation.")

metadata = ["source_file", "bearing_id", "label"]
feature_columns = [
    col for col in healthy.columns if col not in metadata
]

print("=" * 65)
print("CYBERSNIFF - CYBER ATTACK SIMULATOR")
print("=" * 65)
print("Healthy source samples:", len(healthy))
print("Sensor feature columns:", len(feature_columns))

# --------------------------------------------------
# 3. Preserve original healthy samples
# --------------------------------------------------

normal = healthy.copy()
normal["attack_type"] = "None"
normal["label"] = "Normal"

# --------------------------------------------------
# 4. False Data Injection (FDI)
# --------------------------------------------------

fdi = healthy.copy()

# Modify a subset of measurement features.
# The scale is based on each feature's variability.
for col in feature_columns:
    if col.endswith("_mean") or col.endswith("_rms"):
        scale = healthy[col].std()
        if pd.notna(scale) and scale > 0:
            fdi[col] = fdi[col] + rng.normal(
                loc=0.0,
                scale=2.0 * scale,
                size=len(fdi)
            )

fdi["attack_type"] = "False_Data_Injection"
fdi["label"] = "Cyber_Induced_Anomaly"

# --------------------------------------------------
# 5. Sensor bias simulation
# --------------------------------------------------

bias = healthy.copy()

# Apply a fixed offset to selected sensor features.
# This is a simulated reporting bias, not a physical fault.
bias_targets = [
    col for col in feature_columns
    if col.endswith("_mean") or col.endswith("_rms")
]

for col in bias_targets:
    scale = healthy[col].std()
    if pd.notna(scale) and scale > 0:
        bias[col] = bias[col] + 2.5 * scale

bias["attack_type"] = "Sensor_Bias"
bias["label"] = "Cyber_Induced_Anomaly"

# --------------------------------------------------
# 6. Replay-style feature simulation
# --------------------------------------------------

replay = healthy.copy()

# Reorder recorded feature rows to simulate stale/replayed
# feature records. This is NOT raw time-series replay.
permutation = rng.permutation(len(replay))

for col in feature_columns:
    replay[col] = healthy[col].to_numpy()[permutation]

replay["attack_type"] = "Replay_Style_Feature_Shuffle"
replay["label"] = "Cyber_Induced_Anomaly"

# --------------------------------------------------
# 7. Combine datasets
# --------------------------------------------------

combined = pd.concat(
    [normal, fdi, bias, replay],
    ignore_index=True
)

# Keep the provenance of every sample.
combined["sample_id"] = [
    f"CYBER_{i:05d}" for i in range(1, len(combined) + 1)
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
combined.to_csv(OUTPUT_PATH, index=False)

# --------------------------------------------------
# 8. Report results
# --------------------------------------------------

print("\nGenerated class distribution:")
print(combined["label"].value_counts())

print("\nAttack type distribution:")
print(combined["attack_type"].value_counts())

print("\nDataset shape:", combined.shape)
print("\nMissing values:", combined.isnull().sum().sum())

print("\nSaved simulated dataset to:")
print(OUTPUT_PATH)

print("\nSIMULATION COMPLETED!")
