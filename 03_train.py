from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def main():

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # upgrade brain: nano -> small
    model = YOLO("yolov8s.pt")

    model.train(
        data="./visdrone.yaml",

        epochs=35,         
        imgsz=800,          
        batch=16,          
        device=0,

        name="visdrone_upgrade",
        project="./runs",

        # augmentations
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,

        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        optimizer="AdamW",
        lr0=0.001,
        patience=12,

        workers=0,          # Windows fix

        save=True,
        plots=True
    )

    print("Training complete!")
    print("Best model:")
    print("./runs/detect/runs/visdrone_upgrade/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()