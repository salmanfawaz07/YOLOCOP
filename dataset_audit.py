
from pathlib import Path
from collections import Counter
import hashlib
import csv
import cv2

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

DATASET = Path("Traffic_violation")
OUTPUT = Path("outputs")

CLASS_NAMES = {
    0: "bike",
    1: "numberplate",
    2: "tripple riding",
    3: "using mobile",
    4: "with helmet",
    5: "withouthelmet"
}

SPLITS = ["train", "valid", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT.mkdir(exist_ok=True)

# ==========================================
# 2. AUDIT COUNTERS
# ==========================================

class_counts = {
    split: Counter() for split in SPLITS
}

multi_class_images = Counter()
tiny_boxes = []
large_boxes = []
invalid_annotations = []
missing_labels = []
orphan_labels = []
unreadable_images = []
duplicate_images = []

image_hashes = {}
total_images = 0
total_annotations = 0

# ==========================================
# 3. AUDIT EACH DATASET SPLIT
# ==========================================

for split in SPLITS:

    print(f"\n========== {split.upper()} AUDIT ==========")

    image_folder = DATASET / split / "images"
    label_folder = DATASET / split / "labels"

    if not image_folder.exists() or not label_folder.exists():
        print("ERROR: Image or label folder missing!")
        continue

    images = [
        p for p in image_folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    labels = list(label_folder.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing = image_stems - label_stems
    orphan = label_stems - image_stems

    for name in missing:
        missing_labels.append([split, name])

    for name in orphan:
        orphan_labels.append([split, name])

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images without labels:", len(missing))
    print("Labels without images:", len(orphan))

    total_images += len(images)

    for image_path in images:

        label_path = label_folder / (image_path.stem + ".txt")

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            unreadable_images.append(
                [split, image_path.name]
            )
            continue

        height, width = image.shape[:2]

        # Check exact duplicate images using file hash
        digest = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        if digest in image_hashes:
            previous = image_hashes[digest]

            duplicate_images.append([
                previous,
                f"{split}/{image_path.name}"
            ])
        else:
            image_hashes[digest] = (
                f"{split}/{image_path.name}"
            )

        if not label_path.exists():
            continue

        image_classes = set()

        with open(label_path, "r") as file:

            for line_number, line in enumerate(file, 1):

                values = line.split()

                # Detection labels must contain 5 values
                if len(values) != 5:
                    invalid_annotations.append([
                        split,
                        image_path.name,
                        line_number,
                        "Expected 5 values"
                    ])
                    continue

                try:
                    class_id = int(values[0])
                    x, y, w, h = map(float, values[1:])

                except ValueError:
                    invalid_annotations.append([
                        split,
                        image_path.name,
                        line_number,
                        "Non-numeric value"
                    ])
                    continue

                if class_id not in CLASS_NAMES:
                    invalid_annotations.append([
                        split,
                        image_path.name,
                        line_number,
                        "Unknown class ID"
                    ])
                    continue

                if not all(
                    0 <= v <= 1 for v in (x, y, w, h)
                ) or w <= 0 or h <= 0:
                    invalid_annotations.append([
                        split,
                        image_path.name,
                        line_number,
                        "Invalid normalized coordinates"
                    ])
                    continue

                class_counts[split][class_id] += 1
                total_annotations += 1
                image_classes.add(class_id)

                # Bounding-box area as fraction of image area
                area_fraction = w * h

                if area_fraction < 0.001:
                    tiny_boxes.append([
                        split,
                        image_path.name,
                        CLASS_NAMES[class_id],
                        round(area_fraction, 6)
                    ])

                if area_fraction > 0.80:
                    large_boxes.append([
                        split,
                        image_path.name,
                        CLASS_NAMES[class_id],
                        round(area_fraction, 6)
                    ])

        if len(image_classes) > 1:
            multi_class_images[split] += 1

# ==========================================
# 4. PRINT FINAL REPORT
# ==========================================

print("\n========================================")
print("       DATASET QUALITY AUDIT")
print("========================================")

print("\nTotal images:", total_images)
print("Total valid annotations:", total_annotations)

print("\nCLASS DISTRIBUTION")

for class_id, name in CLASS_NAMES.items():

    total = sum(
        class_counts[s][class_id]
        for s in SPLITS
    )

    print(f"{class_id} - {name}: {total}")

print("\nMULTI-CLASS IMAGES")

for split in SPLITS:
    print(split, ":", multi_class_images[split])

print("\nQUALITY ISSUES")

print("Missing labels:", len(missing_labels))
print("Orphan labels:", len(orphan_labels))
print("Invalid annotations:", len(invalid_annotations))
print("Unreadable images:", len(unreadable_images))
print("Exact duplicate images:", len(duplicate_images))
print("Very small boxes:", len(tiny_boxes))
print("Very large boxes:", len(large_boxes))

# ==========================================
# 5. SAVE DETAILED CSV REPORTS
# ==========================================

def save_report(filename, header, rows):

    with open(
        OUTPUT / filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

save_report(
    "tiny_boxes.csv",
    ["Split", "Image", "Class", "Area fraction"],
    tiny_boxes
)

save_report(
    "large_boxes.csv",
    ["Split", "Image", "Class", "Area fraction"],
    large_boxes
)

save_report(
    "invalid_annotations.csv",
    ["Split", "Image", "Line", "Problem"],
    invalid_annotations
)

save_report(
    "missing_labels.csv",
    ["Split", "Image"],
    missing_labels
)

save_report(
    "orphan_labels.csv",
    ["Split", "Label file"],
    orphan_labels
)

save_report(
    "duplicate_images.csv",
    ["First occurrence", "Duplicate"],
    duplicate_images
)

save_report(
    "unreadable_images.csv",
    ["Split", "Image"],
    unreadable_images
)

print("\nCSV reports saved inside outputs/")
print("DATASET AUDIT COMPLETED")
