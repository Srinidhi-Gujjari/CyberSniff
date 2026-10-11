
import os
import re
import glob
import hashlib
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "real", "paderborn")

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True)
)

if not files:
    raise FileNotFoundError(f"No .mat files found in: {DATA_DIR}")

bearing_counts = Counter()
condition_counts = Counter()
hash_locations = defaultdict(list)
unrecognized = []

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

pattern = re.compile(
    r"^(N\d{2}_M\d{2}_F\d{2})_(KA\d{2}|K\d{3})_(\d+)\.mat$",
    re.IGNORECASE
)

print("=" * 70)
print("PADERBORN DATASET INVENTORY")
print("=" * 70)
print(f"Total MATLAB files found: {len(files)}")

for path in files:
    filename = os.path.basename(path)
    match = pattern.match(filename)

    if not match:
        unrecognized.append(path)
        continue

    condition, bearing, measurement_no = match.groups()
    bearing = bearing.upper()
    condition = condition.upper()

    bearing_counts[bearing] += 1
    condition_counts[(bearing, condition)] += 1

    # Hash the file to identify byte-for-byte duplicates.
    digest = file_hash(path)
    hash_locations[digest].append(path)

print("\n1. FILE COUNTS BY BEARING")
for bearing in sorted(bearing_counts):
    print(f"{bearing}: {bearing_counts[bearing]}")

print("\n2. COUNTS BY BEARING AND OPERATING CONDITION")
for (bearing, condition), count in sorted(condition_counts.items()):
    print(f"{bearing:5s} | {condition:13s} | {count:3d}")

duplicate_groups = [
    paths for paths in hash_locations.values() if len(paths) > 1
]

print("\n3. EXACT DUPLICATE FILE CONTENTS")
if duplicate_groups:
    for index, paths in enumerate(duplicate_groups, start=1):
        print(f"\nDuplicate group {index}:")
        for path in paths:
            print(" ", os.path.relpath(path, DATA_DIR))
    print(f"\nDuplicate groups: {len(duplicate_groups)}")
else:
    print("No byte-for-byte duplicate files found.")

print("\n4. UNRECOGNIZED FILENAMES")
if unrecognized:
    for path in unrecognized:
        print(" ", os.path.relpath(path, DATA_DIR))
else:
    print("All filenames matched the expected pattern.")

print("\nInventory validation complete.")
