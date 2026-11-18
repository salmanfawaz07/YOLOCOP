
from pathlib import Path
import shutil

dataset = Path("Traffic_violation")

backup = Path("dataset_backup/helmet_product_images")

class_id = 4
threshold = 0.80

extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

moved = 0

for split in ["train", "valid", "test"]:

    image_folder = dataset / split / "images"
    label_folder = dataset / split / "labels"

    backup_images = backup / split / "images"
    backup_labels = backup / split / "labels"

    backup_images.mkdir(parents=True, exist_ok=True)
    backup_labels.mkdir(parents=True, exist_ok=True)

    for label_file in label_folder.glob("*.txt"):

        suspicious = False

        for line in label_file.read_text().splitlines():

            values = line.split()

            if len(values) != 5:
                continue

            cls = int(values[0])
            x, y, w, h = map(float, values[1:])

            if cls == class_id and w * h > threshold:
                suspicious = True

        if not suspicious:
            continue

        image_path = next(
            (
                image_folder / (label_file.stem + ext)
                for ext in extensions
                if (image_folder / (label_file.stem + ext)).exists()
            ),
            None
        )

        if image_path is None:
            print("Image not found:", label_file.stem)
            continue

        # Confirm that the label contains only class 4
        classes = {
            int(line.split()[0])
            for line in label_file.read_text().splitlines()
            if len(line.split()) == 5
        }

        if classes != {class_id}:
            print("Skipped mixed-class label:", label_file.name)
            continue

        shutil.move(
            str(image_path),
            str(backup_images / image_path.name)
        )

        shutil.move(
            str(label_file),
            str(backup_labels / label_file.name)
        )

        moved += 1

        print("Quarantined:", image_path.name)

print("\nTOTAL IMAGE-LABEL PAIRS QUARANTINED:", moved)
print("Original files preserved in:", backup)
