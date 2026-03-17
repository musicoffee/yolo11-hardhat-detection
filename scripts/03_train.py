import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "scripts", "yolo11n.pt")
DATA_YAML = os.path.join(BASE_DIR, "data", "hardhat.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

def main():
    print("BASE_DIR =", BASE_DIR)
    print("MODEL_PATH =", MODEL_PATH)
    print("DATA_YAML =", DATA_YAML)
    print("RUNS_DIR =", RUNS_DIR)

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,      # 没有 GPU 就改成 "cpu"
        workers=4,
        project=RUNS_DIR,
        name="hardhat_yolo11n",
        pretrained=True,
        cache=False
    )

if __name__ == "__main__":
    main()