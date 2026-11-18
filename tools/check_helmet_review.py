
from pathlib import Path

dataset = Path("Traffic_violation")

for split in ["train", "valid", "test"]:

    label_folder = dataset / split / "labels"

    print(f"\n========== {split.upper()} ==========")

    for label_file in label_folder.glob("*.txt"):

        rows = []

        for line in label_file.read_text().splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            cls = int(values[0])
            x, y, w, h = map(float, values[1:])

            if cls == 4 and w * h > 0.80:
                rows.append((cls, w * h))

        if rows:
            class_ids = [
                int(line.split()[0])
                for line in label_file.read_text().splitlines()
                if len(line.split()) == 5
            ]

            print(
                label_file.stem,
                "| Classes:", class_ids
            )

print("\nREVIEW COMPLETED")
