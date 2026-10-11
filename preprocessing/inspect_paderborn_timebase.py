
import os
import glob
import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "real", "paderborn")

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True)
)

if not files:
    raise FileNotFoundError(f"No .mat files found in {DATA_DIR}")

# Inspect one healthy and one faulty bearing recording.
selected = {}

for path in files:
    name = os.path.basename(path).upper()

    if "_K001_" in name and "healthy" not in selected:
        selected["healthy"] = path

    if "_KA03_" in name and "faulty" not in selected:
        selected["faulty"] = path

    if len(selected) == 2:
        break


def describe(value, depth=0):
    """Print MATLAB metadata structure without dumping signal arrays."""
    if depth > 3:
        print("  " * depth + "(maximum inspection depth)")
        return

    arr = np.asarray(value)
    indent = "  " * depth

    print(
        f"{indent}type={arr.dtype}, shape={arr.shape}"
    )

    if arr.dtype.names:
        for field in arr.dtype.names:
            print(f"{indent}{field}:")
            describe(arr[field], depth + 1)

    elif arr.dtype == object:
        for item in arr.reshape(-1)[:5]:
            describe(item, depth + 1)

    elif arr.size <= 12:
        print(f"{indent}values={arr.reshape(-1).tolist()}")
    else:
        print(f"{indent}values omitted (large array)")


for label, path in selected.items():
    print("\n" + "=" * 65)
    print(f"{label.upper()}: {os.path.basename(path)}")
    print("=" * 65)

    mat = loadmat(path)
    variable_names = [
        key for key in mat if not key.startswith("__")
    ]

    measurement = mat[variable_names[0]]

    print("\nMeasurement X metadata:")
    describe(measurement["X"][0, 0])

    records = measurement["Y"][0, 0].reshape(-1)

    for record in records:
        name = record["Name"]
        name = str(np.asarray(name).reshape(-1)[0])

        if name in {
            "vibration_1",
            "phase_current_1",
            "force",
            "speed",
            "torque",
        }:
            print(f"\nSignal: {name}")

            for field in ("Raster", "XIndex", "DownSampling"):
                print(f"{field}:")
                describe(record[field])

print("\nTimebase inspection complete.")
