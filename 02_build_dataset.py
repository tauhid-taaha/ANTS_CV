import os, glob, random, shutil
from tqdm import tqdm

random.seed(42)

BASE      = './data/visdrone_raw'
YOLO_BASE = './data/yolo'
TRAIN_DIR = None

for root, dirs, _ in os.walk(BASE):
    if 'images' in dirs and 'labels' in dirs:
        TRAIN_DIR = root
        break

all_images = glob.glob(f'{TRAIN_DIR}/images/*.jpg')
all_labels = glob.glob(f'{TRAIN_DIR}/labels/*.txt')
ann_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in all_labels}

for split in ['train', 'val', 'test']:
    os.makedirs(f'{YOLO_BASE}/images/{split}', exist_ok=True)
    os.makedirs(f'{YOLO_BASE}/labels/{split}', exist_ok=True)

random.shuffle(all_images)
n           = len(all_images)
train_imgs  = all_images[:int(n * 0.80)]
val_imgs    = all_images[int(n * 0.80):int(n * 0.95)]
test_imgs   = all_images[int(n * 0.95):]

PERSON_KEEP = {0, 1}
CAR_KEEP    = {3, 4}

def copy_split(img_list, split):
    ok = skipped = 0
    for img_path in tqdm(img_list, desc=f'Copying {split}'):
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = ann_lookup.get(stem)
        if not ann_path: skipped += 1; continue

        new_lines = []
        with open(ann_path) as f:
            for line in f:
                v = line.strip().split()
                if len(v) < 5: continue
                cls = int(v[0])
                if   cls in PERSON_KEEP: new_cls = 0
                elif cls in CAR_KEEP:    new_cls = 1
                else:                    continue
                new_lines.append(f'{new_cls} {v[1]} {v[2]} {v[3]} {v[4]}')

        if not new_lines: skipped += 1; continue

        shutil.copy2(img_path, f'{YOLO_BASE}/images/{split}/{os.path.basename(img_path)}')
        with open(f'{YOLO_BASE}/labels/{split}/{stem}.txt', 'w') as f:
            f.write('\n'.join(new_lines))
        ok += 1
    print(f'  {split}: {ok} saved, {skipped} skipped')

copy_split(train_imgs, 'train')
copy_split(val_imgs,   'val')
copy_split(test_imgs,  'test')
print(' Dataset ready')