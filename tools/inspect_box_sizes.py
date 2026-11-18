
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

dataset = Path("Traffic_violation")

classes = {
    0: "bike",
    1: "numberplate",
    2: "tripple riding",
    3: "using mobile",
    4: "with helmet",
    5: "withouthelmet"
}

small_boxes = []
large_boxes = []

for split in ["train", "valid", "test"]:

    image_folder = dataset / split / "images"
    label_folder = dataset / split / "labels"

    for label_file in label_folder.glob("*.txt"):

        image_path = next(
            (
                image_folder / (label_file.stem + ext)
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                if (image_folder / (label_file.stem + ext)).exists()
            ),
            None
        )

        if image_path is None:
            continue

        with open(label_file) as file:

            for line in file:

                values = line.split()

                if len(values) != 5:
                    continue

                cls, x, y, w, h = map(float, values)

                area = w * h

                item = (
                    image_path,
                    int(cls),
                    x, y, w, h,
                    area
                )

                if area < 0.001:
                    small_boxes.append(item)

                if area > 0.80:
                    large_boxes.append(item)


def display_samples(items, title):

    random.seed(42)
    selected = random.sample(items, min(5, len(items)))

    fig, axes = plt.subplots(
        1, len(selected),
        figsize=(20, 7)
    )

    if len(selected) == 1:
        axes = [axes]

    for ax, item in zip(axes, selected):

        path, cls, x, y, w, h, area = item

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

        name = classes.get(cls, "unknown")

        cv2.putText(
            image, name,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )

        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{name}\nArea: {area:.5f}",
            fontsize=9
        )
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


print("Small boxes:", len(small_boxes))
print("Large boxes:", len(large_boxes))

display_samples(
    small_boxes,
    "Very Small Bounding Boxes"
)

display_samples(
    large_boxes,
    "Very Large Bounding Boxes"
)
