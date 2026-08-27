import json
import random
from pathlib import Path

DATA_DIR = Path("/workspace/US-365K")
IMG_DIR = DATA_DIR / "images"
OUT_DIR = DATA_DIR / "prepared"
OUT_DIR.mkdir(exist_ok=True)

LABEL_FIELDS = [
    "Body_system_level",
    "Organ_level",
    "Diagnosis",
    "Shape",
    "Margins",
    "Echogenicity",
    "InternalCharacteristics",
    "PosteriorAcoustics",
    "Vascularity",
]

print("Building image index...")
image_index = {}
duplicates = 0

for p in IMG_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        name = p.name
        if name in image_index:
            duplicates += 1
        image_index[name] = str(p)

print("Total image files found:", len(image_index))
print("Duplicate image names:", duplicates)

print("Loading full_data.json...")
with open(DATA_DIR / "full_data.json", "r", encoding="utf-8") as f:
    full_data = json.load(f)

if isinstance(full_data, dict):
    if "data" in full_data:
        full_data = full_data["data"]
    else:
        full_data = list(full_data.values())

label_map = {}
for item in full_data:
    name = item.get("media_name") or item.get("image")
    if name:
        label_map[name] = item

print("Total label records:", len(label_map))

def process_split(split_name):
    input_file = DATA_DIR / f"{split_name}.jsonl"
    output_file = OUT_DIR / f"{split_name}_with_paths.jsonl"

    total = 0
    found = 0
    missing = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            item = json.loads(line)

            image_name = item.get("image") or item.get("media_name")
            image_path = image_index.get(image_name)

            if image_path is None:
                missing += 1
                continue

            found += 1
            new_item = {
                "image": image_name,
                "image_path": image_path,
                "caption": item.get("caption", "")
            }

            label_item = label_map.get(image_name, {})
            for field in LABEL_FIELDS:
                new_item[field] = label_item.get(field, [])

            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print(f"{split_name}: total={total}, found={found}, missing={missing}")
    print("Saved:", output_file)

process_split("train")
process_split("valid")
process_split("test")

def create_subset(input_name, n):
    input_file = OUT_DIR / f"{input_name}_with_paths.jsonl"
    output_file = OUT_DIR / f"{input_name}_{n}.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(42)
    if n < len(lines):
        lines = random.sample(lines, n)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("Created subset:", output_file, "samples:", len(lines))

create_subset("train", 1000)
create_subset("train", 5000)
create_subset("train", 10000)
create_subset("train", 50000)
create_subset("valid", 5000)
create_subset("test", 5000)
