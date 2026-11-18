
from pathlib import Path
import yaml

yaml_path = Path("Traffic_violation/data.yaml")

with open(yaml_path, "r") as file:
    data = yaml.safe_load(file)

dataset_path = Path(data["path"])

print("\nDATASET CONFIGURATION")
print("=====================")

print("Dataset:", dataset_path)
print("Number of classes:", data["nc"])
print("Class names:", data["names"])

for split in ["train", "val", "test"]:

    image_path = dataset_path / data[split]

    print(f"\n{split.upper()} PATH:")
    print(image_path)

    if image_path.exists():
        count = len(list(image_path.glob("*")))
        print("STATUS: FOUND")
        print("Files:", count)
    else:
        print("STATUS: PATH NOT FOUND")

print("\nVERIFICATION COMPLETED")
