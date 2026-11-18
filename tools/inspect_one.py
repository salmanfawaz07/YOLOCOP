
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

dataset = Path("Traffic_violation")
image_folder = dataset / "train" / "images"
label_folder = dataset / "train" / "labels"

class_names = {
    0: "bike",
    1: "numberplate",
    2: "tripple riding",
    3: "using mobile",
    4: "with helmet",
    5: "withouthelmet"
}

# Select which class to inspect
print("\nSELECT CLASS TO INSPECT")
for class_id, name in class_names.items():
    print(f"{class_id} - {name}")

class_id = int(input("\nEnter class ID: "))

if class_id not in class_names:
    raise ValueError("Invalid class ID")

# Find images containing the selected class
matching_images = []

for label_file in label_folder.glob("*.txt"):
    lines = label_file.read_text().splitlines()

    if any(
        line.split() and line.split()[0] == str(class_id)
        for line in lines
    ):
        matching_images.append(label_file.stem)

# Match label files with images
images = [
    image_folder / (name + ext)
    for name in matching_images
    for ext in [".jpg", ".jpeg", ".png"]
    if (image_folder / (name + ext)).exists()
]

print("\nClass:", class_names[class_id])
print("Images containing this class:", len(images))

if not images:
    raise SystemExit("No matching images found")

random.seed(42)
selected = random.sample(images, min(5, len(images)))

# Display five matching images
fig, axes = plt.subplots(1, len(selected), figsize=(20, 7))

if len(selected) == 1:
    axes = [axes]

for ax, image_path in zip(axes, selected):

    label_path = label_folder / (image_path.stem + ".txt")
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    for line in label_path.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        cls = int(values[0])
        x, y, w, h = map(float, values[1:])

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        cv2.rectangle(
            image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            image, class_names.get(cls, "unknown"),
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(image_path.name[:25])
    ax.axis("off")

plt.tight_layout()
plt.show()
