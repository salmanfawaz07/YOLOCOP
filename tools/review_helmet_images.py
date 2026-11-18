
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

dataset = Path("Traffic_violation")

suspicious = []

for split in ["train", "valid", "test"]:

    image_folder = dataset / split / "images"
    label_folder = dataset / split / "labels"

    for label_file in label_folder.glob("*.txt"):

        for line in label_file.read_text().splitlines():

            values = line.split()

            if len(values) != 5:
                continue

            cls, x, y, w, h = map(float, values)

            if int(cls) == 4 and w * h > 0.80:

                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    image_path = image_folder / (label_file.stem + ext)

                    if image_path.exists():
                        suspicious.append(
                            (split, image_path, x, y, w, h)
                        )
                        break

print("Images found:", len(suspicious))

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for ax, item in zip(axes, suspicious):

    split, path, x, y, w, h = item

    image = cv2.imread(str(path))
    H, W = image.shape[:2]

    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)

    cv2.rectangle(
        image, (x1, y1), (x2, y2),
        (0, 255, 0), 2
    )

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{split}: {path.name[:23]}", fontsize=8)
    ax.axis("off")

for ax in axes[len(suspicious):]:
    ax.axis("off")

plt.suptitle("Review: Suspicious With-Helmet Annotations")
plt.tight_layout()
plt.show()
