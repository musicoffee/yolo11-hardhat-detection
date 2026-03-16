from ultralytics import YOLO

def main():

    model = YOLO("yolo11n.pt")

    model.train(
        data=r"F:\yolo\data\hardhat.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=r"F:\yolo\runs",
        name="hardhat_yolo11n",
        pretrained=True,
        cache=False
    )

if __name__ == "__main__":
    main()