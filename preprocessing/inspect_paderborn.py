
import os
from scipy.io import loadmat


# ============================================
# CYBERSNIFF - PADERBORN DATA INSPECTOR
# ============================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_directory = os.path.join(
    project_root,
    "data",
    "real",
    "paderborn",
    "K001",
    "K001"
)

filename = "N09_M07_F10_K001_1.mat"

file_path = os.path.join(
    dataset_directory,
    filename
)

print("=" * 60)
print("CYBERSNIFF - PADERBORN DATA INSPECTION")
print("=" * 60)

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        "Check the dataset folder path."
    )

print("\nReading file:")
print(file_path)

mat_data = loadmat(file_path)

print("\nAvailable MATLAB variables:")
print("-" * 60)

for key, value in mat_data.items():

    # MATLAB metadata keys begin with __
    if key.startswith("__"):
        continue

    print(f"\nVariable name: {key}")
    print(f"Data type: {type(value)}")
    print(f"Shape: {getattr(value, 'shape', 'N/A')}")

    if hasattr(value, "dtype"):
        print(f"Data dtype: {value.dtype}")

    # Show a small preview for numeric arrays
    if hasattr(value, "shape") and value.size > 0:
        print("Sample values:")
        print(value.reshape(-1)[:5])

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)
