from ultralytics import YOLO
from pathlib import Path

def main():
    dataset_root = Path(r".\data\cabinet_dataset")

    model = YOLO("yolov8m-cls.pt")

    results = model.train(
        data=str(dataset_root),
        epochs=50,
        imgsz=224,
        batch=32,
        lr0=1e-3,
        device=0,
        patience=10,
        workers=1,
    )

    print("Training complete. Results:", results)

if __name__ == "__main__":
    main()