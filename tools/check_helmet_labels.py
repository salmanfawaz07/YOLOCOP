
from pathlib import Path

dataset = Path("Traffic_violation")

class_id_to_check = 4
class_name = "with helmet"

found = 0

for split in ["train", "valid", "test"]:

    label_folder = dataset / split / "labels"

    print(f"\n========== {split.upper()} ==========")

    for label_file in label_folder.glob("*.txt"):

        for line_number, line in enumerate(
            label_file.read_text().splitlines(), 1
        ):

            values = line.split()

            if len(values) != 5:
                continue

            try:
                cls = int(values[0])
                x, y, w, h = map(float, values[1:])
            except ValueError:
                continue

            area = w * h

            if cls == class_id_to_check and area > 0.80:

                found += 1

                print(
                    "Image:", label_file.stem,
                    "| Area:", round(area, 4),
                    "| Line:", line_number
                )

print("\nTotal suspicious helmet boxes:", found)
