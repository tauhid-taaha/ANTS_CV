import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import Counter

random.seed(42)
np.random.seed(42)

BASE      = './data/visdrone_raw'
TRAIN_DIR = None

for root, dirs, _ in os.walk(BASE):
    if 'images' in dirs and 'labels' in dirs:
        TRAIN_DIR = root
        break

print(f'Train dir: {TRAIN_DIR}')

all_images = glob.glob(f'{TRAIN_DIR}/images/*.jpg')
all_labels = glob.glob(f'{TRAIN_DIR}/labels/*.txt')
ann_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in all_labels}

CLASS_NAMES = {
    0:'pedestrian', 1:'people',    2:'bicycle',
    3:'car',        4:'van',       5:'truck',
    6:'tricycle',   7:'awning-tricycle', 8:'bus', 9:'motor'
}

class_counter = Counter()
obj_per_img   = []
bbox_sizes    = []

for img_path in tqdm(random.sample(all_images, min(500, len(all_images))), desc='EDA'):
    stem     = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = ann_lookup.get(stem)
    if not ann_path: continue
    count = 0
    with open(ann_path) as f:
        for line in f:
            v = line.strip().split()
            if len(v) < 5: continue
            try:
                cls = int(v[0])
                w, h = float(v[3]), float(v[4])
                class_counter[cls] += 1
                bbox_sizes.append((w, h))
                count += 1
            except: pass
    obj_per_img.append(count)

print(f'Classes found : {dict(sorted(class_counter.items()))}')
print(f'Avg objects   : {np.mean(obj_per_img):.1f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('VisDrone EDA', fontsize=14, fontweight='bold')

cats   = [CLASS_NAMES.get(k, str(k)) for k in sorted(class_counter.keys())]
cnts   = [class_counter[k] for k in sorted(class_counter.keys())]
colors = ['#e74c3c' if c in ['pedestrian','people']
          else '#3498db' if c in ['car','van']
          else '#95a5a6' for c in cats]
axes[0].barh(cats, cnts, color=colors)
axes[0].set_title('Class Distribution')

axes[1].hist(obj_per_img, bins=30, color='#2ecc71', edgecolor='white')
axes[1].axvline(np.mean(obj_per_img), color='red', linestyle='--')
axes[1].set_title('Objects per Image')

bw = [b[0] for b in bbox_sizes[:1000]]
bh = [b[1] for b in bbox_sizes[:1000]]
axes[2].scatter(bw, bh, alpha=0.3, s=5, color='#e74c3c')
axes[2].axvline(0.05, color='orange', linestyle='--')
axes[2].axhline(0.05, color='orange', linestyle='--')
axes[2].set_xlim(0, 0.3); axes[2].set_ylim(0, 0.3)
axes[2].set_title('BBox Sizes (normalised)')

plt.tight_layout()
os.makedirs('./outputs', exist_ok=True)
plt.savefig('./outputs/eda.png', dpi=150)
plt.show()
print(' EDA saved → outputs/eda.png')