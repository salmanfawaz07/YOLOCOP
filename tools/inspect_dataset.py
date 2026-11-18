
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# Dataset folders
dataset = Path("Traffic_violation")
image_folder = dataset / "train" / "images"
label_folder = dataset / "train" / "labels"

# Dataset classes
class_names = {
    0: "bike",
    1: "numberplate",
    2: "tripple riding",
    3: "using mobile",
    4: "with helmet",
    5: "withouthelmet"
}

# Verify folders
if not image_folder.exists() or not label_folder.exists():
    raise FileNotFoundError("Training images or labels folder not found.")

# Find images
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
images = [
    p for p in image_folder.iterdir()
    if p.suffix.lower() in extensions
]

print("Training images found:", len(images))

# Select 12 random images
random.seed(42)
selected = random.sample(images, min(12, len(images)))

print("Images selected:", len(selected))

# Display images in a grid
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()

for index, image_path in enumerate(selected):

    ax = axes[index]
    label_path = label_folder / (image_path.stem + ".txt")

    image = cv2.imread(str(image_path))

    if image is None:
        ax.set_title("Image could not be opened")
        ax.axis("off")
        continue

    height, width = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r") as file:

            for line in file:
                values = line.split()

                # Detection format: class + 4 coordinates
                if len(values) != 5:
                    continue

                class_id = int(values[0])
                x, y, w, h = map(float, values[1:])

                # YOLO coordinates to pixel coordinates
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)

                # Draw bounding box
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Class label
                name = class_names.get(class_id, "unknown")

                cv2.putText(
                    image,
                    name,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2
                )

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(image_path.name, fontsize=9)
    ax.axis("off")

# Hide unused grid cells
for index in range(len(selected), len(axes)):
    axes[index].axis("off")

plt.suptitle(
    "Traffic Violation Dataset - Annotation Inspection",
    fontsize=18
)

plt.tight_layout()
plt.show()

print("Visual inspection completed.")
