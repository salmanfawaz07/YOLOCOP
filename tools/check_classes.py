
from pathlib import Path
from collections import Counter

# Dataset location
dataset = Path("Traffic_violation")

# Class names from data.yaml
class_names = {
    0: "bike",
    1: "numberplate",
    2: "tripple riding",
    3: "using mobile",
    4: "with helmet",
    5: "withouthelmet"
}

splits = ["train", "valid", "test"]

for split in splits:

    label_folder = dataset / split / "labels"

    print(f"\n========== {split.upper()} ==========")

    if not label_folder.exists():
        print("Labels folder not found:", label_folder)
        continue

    counts = Counter()
    total = 0
    invalid = 0

    for label_file in label_folder.glob("*.txt"):

        with open(label_file, "r") as file:

            for line in file:

                values = line.split()

                if len(values) < 5:
                    invalid += 1
                    continue

                try:
                    class_id = int(values[0])
                except ValueError:
                    invalid += 1
                    continue

                if class_id not in class_names:
                    invalid += 1
                    continue

                counts[class_id] += 1
                total += 1

    print("Total annotations:", total)

    for class_id, name in class_names.items():
        print(f"{name}: {counts[class_id]}")

    print("Invalid rows:", invalid)

print("\nANNOTATION CHECK COMPLETED")
