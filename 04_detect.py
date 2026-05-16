import os, glob, random, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('./runs/detect/runs/visdrone_upgrade/weights/best.pt')
os.makedirs('./outputs', exist_ok=True)

test_images = glob.glob('./data/yolo/images/test/*.jpg')
samples     = random.sample(test_images, min(8, len(test_images)))

def detect_and_count(img_path, model, conf=0.10, iou=0.35):
    t0 = time.time()

    res = model.predict(
        img_path,
        imgsz=960,          # bigger image -> tiny cars become larger
        conf=conf,          # lower threshold -> keep more detections
        iou=iou,            # slightly more permissive
        augment=True,       # test-time augmentation
        device=0,
        verbose=False
    )[0]

    ms=(time.time()-t0)*1000

    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    persons=0
    cars=0

    for box in res.boxes:

        cls=int(box.cls[0])
        conf_=float(box.conf[0])

        # remove extremely weak predictions
        if conf_ < 0.08:
            continue

        x1,y1,x2,y2=map(int,box.xyxy[0])

        if cls==0:
            color=(255,60,60)
            persons+=1
            label=f'P {conf_:.2f}'

        elif cls==1:
            color=(60,140,255)
            cars+=1
            label=f'C {conf_:.2f}'

        else:
            continue

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            img,
            label,
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,255,255),
            1
        )

    H,W=img.shape[:2]

    cv2.rectangle(img,(0,0),(380,45),(0,0,0),-1)

    cv2.putText(img,f'Persons:{persons}',
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,100,100),
                2)

    cv2.putText(img,f'Cars:{cars}',
                (180,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100,180,255),
                2)

    return img,persons,cars,ms

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('Detection Results | Red=Person  Blue=Car', fontsize=13, fontweight='bold')

total_persons = total_cars = total_ms = 0
for i, p in enumerate(samples):
    out_img, pc, cc, ms = detect_and_count(p, model)
    total_persons += pc; total_cars += cc; total_ms += ms
    ax = axes[i // 4][i % 4]
    ax.imshow(out_img)
    ax.set_title(f'P:{pc}  C:{cc}  ({ms:.0f}ms)')
    ax.axis('off')

plt.tight_layout()
plt.savefig('./outputs/detection_results.png', dpi=150)
plt.show()
print(f'Total persons: {total_persons} | Total cars: {total_cars}')
print(f'Avg FPS: {1000/(total_ms/len(samples)):.1f}')
print(' Saved → outputs/detection_results.png')