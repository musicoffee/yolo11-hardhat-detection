import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "runs", "hardhat_yolo11n", "weights", "best.pt")
SOURCE_DIR = os.path.join(BASE_DIR, "data", "images", "test")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

def main():
    model = YOLO(MODEL_PATH)

    model.predict(
        source=SOURCE_DIR,
        imgsz=640,
        conf=0.25,
        save=True,
        project=RUNS_DIR,
        name="hardhat_predict"
    )

    print("预测完成。")

if __name__ == "__main__":
    main()