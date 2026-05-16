from ultralytics import YOLO
from multiprocessing import freeze_support

def main():

    model = YOLO('./runs/detect/runs/visdrone_upgrade/weights/best.pt')

    results = model.val(
        data='./visdrone.yaml',
        split='test',

        imgsz=960,        # larger resolution for tiny drone objects
        conf=0.10,        # keep weaker detections
        iou=0.45,         # slightly friendlier matching
        augment=True,     # test-time augmentation
        agnostic_nms=True,
        max_det=1000,

        batch=16,
        workers=0,        # Windows fix
        device=0,

        verbose=True,
        plots=True
    )

    print('\n' + '='*50)
    print('     FINAL TEST RESULTS')
    print('='*50)

    print(f'Precision    : {results.box.mp:.4f}')
    print(f'Recall       : {results.box.mr:.4f}')
    print(f'mAP @0.5     : {results.box.map50:.4f}')
    print(f'mAP @.5:.95  : {results.box.map:.4f}')

    if len(results.box.maps) >= 2:
        print(f'Person mAP   : {results.box.maps[0]:.4f}')
        print(f'Car mAP      : {results.box.maps[1]:.4f}')

    print('='*50)

if __name__ == "__main__":
    freeze_support()
    main()