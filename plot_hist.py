from collections import Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm
from collections import defaultdict
import os
import seaborn as sns

# --- Path ---
image_dir = "dataset/FLIR_Aligned/images_rgb_train/data"
ann_path = "dataset/FLIR_Aligned/meta/rgb/flir_train.json"

coco = COCO(ann_path)
cats = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

pixel_values = {
    "RGB": defaultdict(list),
    "Thermal": defaultdict(list)
}

for ann_id in tqdm(coco.getAnnIds()):
    ann = coco.loadAnns(ann_id)[0]
    cat_id = ann["category_id"]
    image_id = ann["image_id"]
    image_info = coco.loadImgs([image_id])[0]

    img_path = os.path.join(image_dir, image_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = map(int, ann["bbox"])
    crop = rgb_img[y:y + h, x:x + w]
    try:
      norm = crop.astype(np.float32) / 255.0
      pixel_values['RGB'][cat_id].extend(norm.flatten())
    except:
      continue

image_dir = "dataset/FLIR_Aligned/images_thermal_train/data"
ann_path = "dataset/FLIR_Aligned/meta/thermal/flir_train.json"

coco = COCO(ann_path)
cats = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

for ann_id in tqdm(coco.getAnnIds()):
    ann = coco.loadAnns(ann_id)[0]
    cat_id = ann["category_id"]
    image_id = ann["image_id"]
    image_info = coco.loadImgs([image_id])[0]

    img_path = os.path.join(image_dir, image_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = map(int, ann["bbox"])
    crop = rgb_img[y:y + h, x:x + w]
    try:
      norm = crop.astype(np.float32) / 255.0
      pixel_values['Thermal'][cat_id].extend(norm.flatten())
    except:
      continue


for cat_id, pixels in pixel_values['RGB'].items():
    cat_name = cat_id_to_name[cat_id]
    filtered_pixels = [p for p in pixels if p > 1e-5]
    plt.hist(filtered_pixels, bins=256, alpha=0.6, label=cat_name, density=True)

plt.title("Pixel Intensity Distribution per Class in FLIR RGB Images")
plt.xlabel("Pixel Intensity")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()



class_counts = Counter()
for ann_id in coco.getAnnIds():
    ann = coco.loadAnns(ann_id)[0]
    cat_id = ann["category_id"]
    class_counts[cat_id] += 1

selected_classes = ["person", "car", "bike"]
labels = []
counts = []
cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
for class_name in selected_classes:
    if class_name not in cat_name_to_id:
        continue
    cat_id = cat_name_to_id[class_name]
    labels.append(class_name.capitalize())
    counts.append(class_counts.get(cat_id, 0))

total = sum(counts)
percentages = [c / total * 100 for c in counts]

colors = {
    "Person": "#ff6f61",
    "Car": "#6baed6",
    "Bicycle": "#74c476"
}
bar_colors = [colors.get(label, "#cccccc") for label in labels]

sns.set(style="whitegrid")
plt.figure(figsize=(9, 6))
bars = plt.bar(labels, counts, color=bar_colors, linewidth=1.2)

for i, (count, pct) in enumerate(zip(counts, percentages)):
    plt.text(i, count + total * 0.02, f"{count} ({pct:.1f}%)",
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Class Distribution in FLIR Annotations", fontsize=16)
plt.xlabel("Class", fontsize=13)
plt.ylabel("Number of Annotations", fontsize=13)
plt.ylim(0, max(counts) * 1.2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()