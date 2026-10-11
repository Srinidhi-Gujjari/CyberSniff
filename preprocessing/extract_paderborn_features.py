
import os
import glob
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import kurtosis


# ============================================================
# CYBERSNIFF - PADERBORN FEATURE EXTRACTION
# ============================================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_root = os.path.join(
    project_root,
    "data",
    "real",
    "paderborn",
    "K001"
)

output_directory = os.path.join(
    project_root,
    "data",
    "processed"
)

os.makedirs(output_directory, exist_ok=True)


# ============================================================
# HELPER: EXTRACT TEXT FROM MATLAB STRUCTURES
# ============================================================

def matlab_text(value):
    """Convert a MATLAB string-like value into Python text."""

    array = np.asarray(value)

    if array.size == 0:
        return ""

    item = array.reshape(-1)[0]

    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")

    if isinstance(item, str):
        return item

    return str(item)


# ============================================================
# EXTRACT FEATURES FROM ONE MATLAB FILE
# ============================================================

def extract_features(mat_file):

    mat_data = loadmat(mat_file)

    # Locate the main MATLAB variable.
    main_keys = [
        key for key in mat_data
        if not key.startswith("__")
    ]

    if not main_keys:
        raise ValueError("No MATLAB measurement variable found.")

    measurement = mat_data[main_keys[0]]

    # The main structure contains a Y field with signal records.
    signal_records = measurement["Y"][0, 0]

    features = {
        "source_file": os.path.basename(mat_file)
    }

    # MATLAB Y is a structured array of signal records.
    for record in signal_records.reshape(-1):

        signal_name = matlab_text(record["Name"])

        if not signal_name:
            continue

        signal_data = np.asarray(
            record["Data"],
            dtype=np.float64
        ).reshape(-1)

        # Ignore empty or invalid signals.
        signal_data = signal_data[
            np.isfinite(signal_data)
        ]

        if signal_data.size == 0:
            continue

        # Use descriptive names, not misleading pump equivalents.
        prefix = signal_name

        features[f"{prefix}_mean"] = np.mean(signal_data)

        features[f"{prefix}_std"] = np.std(signal_data)

        features[f"{prefix}_rms"] = np.sqrt(
            np.mean(signal_data ** 2)
        )

        features[f"{prefix}_min"] = np.min(signal_data)

        features[f"{prefix}_max"] = np.max(signal_data)

        features[f"{prefix}_peak_to_peak"] = (
            np.max(signal_data) - np.min(signal_data)
        )

        features[f"{prefix}_kurtosis"] = (
            kurtosis(signal_data, fisher=True, bias=False)
            if signal_data.size > 3
            else np.nan
        )

    return features


# ============================================================
# MAIN: INSPECT ONE FILE FIRST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("CYBERSNIFF - PADERBORN FEATURE EXTRACTION")
    print("=" * 65)

    mat_files = sorted(
        glob.glob(
            os.path.join(dataset_root, "**", "*.mat"),
            recursive=True
        )
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found under: {dataset_root}"
        )

    # Start with one file to verify the extraction.
    selected_file = mat_files[0]

    print("\nSelected file:")
    print(selected_file)

    features = extract_features(selected_file)

    feature_df = pd.DataFrame([features])

    print("\nExtracted features:")
    print(feature_df.T.to_string())

    output_file = os.path.join(
        output_directory,
        "paderborn_feature_preview.csv"
    )

    feature_df.to_csv(output_file, index=False)

    print("\nFeature preview saved to:")
    print(output_file)

    print("\nExtraction completed successfully.")
