
import os
import glob
import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "real", "paderborn")

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True)
)

# Inspect one healthy and one faulty example if available.
selected = []
for path in files:
    name = os.path.basename(path).upper()

    if "_K001_" in name or "_KA03_" in name:
        selected.append(path)

    if len(selected) >= 2:
        break

if not selected:
    selected = files[:2]

for path in selected:
    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(path))
    print("=" * 70)

    mat = loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]
    print("MATLAB variables:", keys)

    if not keys:
        continue

    measurement = mat[keys[0]]
    print("Measurement dtype:", getattr(measurement, "dtype", None))

    try:
        signal_records = measurement["Y"][0, 0]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        print("Could not access measurement Y:", error)
        continue

    for record in signal_records.reshape(-1):
        fields = record.dtype.names or ()
        print("\nRecord fields:", fields)

        try:
            name = record["Name"]
            if isinstance(name, np.ndarray):
                name = name.reshape(-1)[0]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            print("Signal name:", name)
        except (ValueError, KeyError, IndexError, TypeError):
            print("Signal name unavailable")

        try:
            values = np.asarray(record["Data"])
            print("Data shape:", values.shape)
            print("Data dtype:", values.dtype)
            print("Number of values:", values.size)
        except (ValueError, KeyError, IndexError, TypeError):
            print("Signal data unavailable")

print("\nInspection complete.")
