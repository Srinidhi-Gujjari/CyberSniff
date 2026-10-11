
import os
import glob
import hashlib

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import kurtosis


# ============================================================
# CYBERSNIFF - PADERBORN DATASET EXTRACTION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "data", "real", "paderborn"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "paderborn_real_features.csv"
)

ERROR_FILE = os.path.join(
    OUTPUT_DIR, "paderborn_extraction_errors.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# BEARING LABELS
# NOTE: Verify exact fault subtypes against official fact sheets.
# The binary label is the target for the first classifier.
# ============================================================

BEARING_LABELS = {
    "K001": {
        "label": "Healthy",
        "damage_type": "Not_Applicable"
    },
    "K002": {
        "label": "Healthy",
        "damage_type": "Not_Applicable"
    },
    "K003": {
        "label": "Healthy",
        "damage_type": "Not_Applicable"
    },
    "KA03": {
        "label": "Physical_Fault",
        "damage_type": "Outer_Race_Damage"
    },
    "KA04": {
        "label": "Physical_Fault",
        "damage_type": "Outer_Race_Damage"
    },
    "KA05": {
        "label": "Physical_Fault",
        "damage_type": "Outer_Race_Damage"
    }
}


# ============================================================
# DUPLICATE DETECTION
# Skip byte-identical copies. Never delete source files.
# ============================================================

def sha256_file(file_path):
    """Calculate SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_unique_files(file_paths):
    """Keep one path for each unique file content."""

    unique_files = []
    seen_hashes = set()
    duplicate_files = []

    for file_path in file_paths:
        digest = sha256_file(file_path)

        if digest in seen_hashes:
            duplicate_files.append(file_path)
        else:
            seen_hashes.add(digest)
            unique_files.append(file_path)

    print(f"Files discovered: {len(file_paths)}")
    print(f"Unique file contents: {len(unique_files)}")
    print(f"Duplicate copies skipped: {len(duplicate_files)}")

    if duplicate_files:
        print("\nDuplicate paths skipped:")
        for path in duplicate_files:
            print("  ", os.path.relpath(path, DATASET_ROOT))

    return unique_files


# ============================================================
# MATLAB TEXT CONVERSION
# ============================================================

def matlab_text(value):
    """Convert a MATLAB character value to a Python string."""

    array = np.asarray(value)

    if array.size == 0:
        return ""

    item = array.reshape(-1)[0]

    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace").strip()

    return str(item).strip()


# ============================================================
# IDENTIFY BEARING
# ============================================================

def identify_bearing(mat_file):
    """Identify bearing ID from filename or parent folders."""

    filename = os.path.basename(mat_file).upper()

    for bearing_id in sorted(
        BEARING_LABELS, key=len, reverse=True
    ):
        if f"_{bearing_id}_" in filename:
            return bearing_id

    path_parts = [
        part.upper()
        for part in os.path.normpath(mat_file).split(os.sep)
    ]

    for bearing_id in sorted(
        BEARING_LABELS, key=len, reverse=True
    ):
        if bearing_id in path_parts:
            return bearing_id

    return None


# ============================================================
# SENSOR FEATURE EXTRACTION
# ============================================================

def extract_features(mat_file):
    """Extract statistical features from one MATLAB recording."""

    mat_data = loadmat(mat_file)

    main_keys = [
        key for key in mat_data
        if not key.startswith("__")
    ]

    if not main_keys:
        raise ValueError("No measurement variable found.")

    measurement = mat_data[main_keys[0]]

    try:
        signal_records = measurement["Y"][0, 0]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            "Could not locate measurement sensor records."
        ) from error

    features = {
        "source_file": os.path.basename(mat_file)
    }

    for record in signal_records.reshape(-1):
        signal_name = matlab_text(record["Name"])

        if not signal_name:
            continue

        values = np.asarray(
            record["Data"],
            dtype=np.float64
        ).reshape(-1)

        # Ignore NaN and infinite samples inside a signal.
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        prefix = signal_name

        features[f"{prefix}_mean"] = float(
            np.mean(values)
        )

        features[f"{prefix}_std"] = float(
            np.std(values)
        )

        features[f"{prefix}_rms"] = float(
            np.sqrt(np.mean(values ** 2))
        )

        features[f"{prefix}_min"] = float(
            np.min(values)
        )

        features[f"{prefix}_max"] = float(
            np.max(values)
        )

        features[f"{prefix}_peak_to_peak"] = float(
            np.ptp(values)
        )

        if values.size > 3:
            features[f"{prefix}_kurtosis"] = float(
                kurtosis(
                    values,
                    fisher=True,
                    bias=False
                )
            )
        else:
            features[f"{prefix}_kurtosis"] = np.nan

    if len(features) == 1:
        raise ValueError(
            "No usable sensor features were extracted."
        )

    return features


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():

    print("=" * 70)
    print("CYBERSNIFF - PADERBORN REAL-DATA EXTRACTION")
    print("=" * 70)

    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    all_mat_files = sorted(
        glob.glob(
            os.path.join(DATASET_ROOT, "**", "*.mat"),
            recursive=True
        )
    )

    if not all_mat_files:
        raise FileNotFoundError(
            "No MATLAB files found in the Paderborn dataset."
        )

    # Skip exact duplicate copies.
    mat_files = get_unique_files(all_mat_files)

    selected_files = []

    for file_path in mat_files:
        bearing_id = identify_bearing(file_path)

        if bearing_id is None:
            print(
                "WARNING: Unrecognized bearing:",
                os.path.basename(file_path)
            )
            continue

        bearing_info = BEARING_LABELS[bearing_id]

        selected_files.append(
            (
                file_path,
                bearing_id,
                bearing_info["label"],
                bearing_info["damage_type"]
            )
        )

    if not selected_files:
        raise RuntimeError(
            "No recognized bearing files found."
        )

    print(
        "\nRecognized measurement files:",
        len(selected_files)
    )

    rows = []
    errors = []

    for index, (
        file_path,
        bearing_id,
        label,
        damage_type
    ) in enumerate(selected_files, start=1):

        try:
            features = extract_features(file_path)

            features["bearing_id"] = bearing_id
            features["label"] = label
            features["damage_type"] = damage_type

            rows.append(features)

        except Exception as error:
            errors.append({
                "file": file_path,
                "error": str(error)
            })

            print(
                f"ERROR [{index}]: "
                f"{os.path.basename(file_path)}: {error}"
            )

        if index % 10 == 0 or index == len(selected_files):
            print(
                f"Processed {index}/{len(selected_files)} files"
            )

    if not rows:
        raise RuntimeError(
            "No measurement files could be processed."
        )

    dataset = pd.DataFrame(rows)

    metadata_columns = [
        "source_file",
        "bearing_id",
        "label",
        "damage_type"
    ]

    feature_columns = [
        column for column in dataset.columns
        if column not in metadata_columns
    ]

    dataset = dataset[
        metadata_columns + feature_columns
    ]

    # Check numerical values before writing the CSV.
    numeric_data = dataset.select_dtypes(
        include=[np.number]
    )

    infinite_count = int(
        np.isinf(numeric_data.to_numpy()).sum()
    )

    # Save extracted data.
    dataset.to_csv(OUTPUT_FILE, index=False)

    # Write errors only when errors exist.
    if errors:
        pd.DataFrame(errors).to_csv(
            ERROR_FILE,
            index=False
        )
    elif os.path.exists(ERROR_FILE):
        os.remove(ERROR_FILE)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"\nSuccessfully processed: {len(dataset)}")
    print(f"Failed files: {len(errors)}")

    print("\nSamples per bearing:")
    print(
        dataset["bearing_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nClass distribution:")
    print(dataset["label"].value_counts().to_string())

    print("\nDamage-type distribution:")
    print(
        dataset["damage_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDataset shape:", dataset.shape)
    print(
        "Missing values:",
        int(dataset.isna().sum().sum())
    )
    print("Infinite numerical values:", infinite_count)

    print("\nCounts by bearing and class:")
    print(
        pd.crosstab(
            dataset["bearing_id"],
            dataset["label"]
        ).to_string()
    )

    print("\nSaved dataset:")
    print(OUTPUT_FILE)

    if errors:
        print("\nExtraction errors saved to:")
        print(ERROR_FILE)

    print("\nEXTRACTION COMPLETED!")


if __name__ == "__main__":
    main()
