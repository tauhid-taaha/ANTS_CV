import os, glob
import cv2
from ultralytics import YOLO

model = YOLO('./runs/detect/runs/visdrone_upgrade/weights/best.pt')
os.makedirs('./outputs', exist_ok=True)

# Build test video from images
test_imgs = sorted(glob.glob('./data/yolo/images/test/*.jpg'))[:80]

VIDEO_IN = './outputs/test_sequence.avi'

sample_frame = cv2.imread(test_imgs[0])
H, W = sample_frame.shape[:2]

writer = cv2.VideoWriter(
    VIDEO_IN,
    cv2.VideoWriter_fourcc(*'XVID'),
    5,
    (W, H)
)

for p in test_imgs:
    fr = cv2.imread(p)

    if fr is None:
        continue

    fr = cv2.resize(fr, (W, H))
    writer.write(fr)

writer.release()
cv2.destroyAllWindows()

print(f'✅ Test video created → {VIDEO_IN}')

# Tracking
results = model.track(
    source=VIDEO_IN,
    tracker='bytetrack.yaml',

    imgsz=960,          # larger input for tiny drone objects
    conf=0.20,          # keep weaker detections
    iou=0.25,
    augment=True,
    agnostic_nms=True,
    max_det=1000,
    persist=True,
    vid_stride=1,

    save=True,
    project='./outputs/tracking',
    name='bytetrack_out',
    device=0,
    stream=True
)

# force generator execution
for r in results:
    pass

print("✅ Tracking complete")
print("Saved in: ./outputs/tracking/bytetrack_out/")