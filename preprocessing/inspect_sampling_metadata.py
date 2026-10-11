
import os
import glob
import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "real", "paderborn")

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True)
)

wanted = {"K001": None, "KA03": None}

for path in files:
    filename = os.path.basename(path).upper()

    for bearing_id in wanted:
        if f"_{bearing_id}_" in filename and wanted[bearing_id] is None:
            wanted[bearing_id] = path

    if all(wanted.values()):
        break


def show_value(value):
    """Print MATLAB metadata in a readable form."""
    arr = np.asarray(value)

    if arr.size == 0:
        return "(empty)"

    values = arr.reshape(-1)
    if len(values) > 8:
        values = values[:8]

    return repr(values.tolist())


for bearing_id, path in wanted.items():
    print("\n" + "=" * 70)
    print(f"BEARING: {bearing_id}")
    print("FILE:", os.path.basename(path) if path else "NOT FOUND")
    print("=" * 70)

    if path is None:
        continue

    mat = loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]

    if not keys:
        print("No MATLAB measurement variable found.")
        continue

    measurement = mat[keys[0]]
    records = measurement["Y"][0, 0].reshape(-1)

    for record in records:
        name = show_value(record["Name"])
        print(f"\nSignal name: {name}")
        print("Data shape:", np.asarray(record["Data"]).shape)

        for field in ["Unit", "Raster", "DownSampling", "Type", "Device"]:
            try:
                print(f"{field}: {show_value(record[field])}")
            except (ValueError, KeyError, IndexError, TypeError):
                print(f"{field}: unavailable")

print("\nMetadata inspection complete.")
