from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "scripts", "yolo11n.pt")
DATA_YAML = os.path.join(BASE_DIR, "data", "hardhat.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

def main():

    def main():
        model = YOLO(MODEL_PATH)

        model.train(
            data=DATA_YAML,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,
            workers=4,
            project=RUNS_DIR,
            name="hardhat_yolo11n",
            pretrained=True,
            cache=False
        )

    if __name__ == "__main__":
        main()