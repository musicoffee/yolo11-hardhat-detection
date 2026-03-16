from ultralytics import YOLO

def main():
    model = YOLO(r"F:\yolo\runs\hardhat_yolo11n\weights\best.pt")

    # 这里 source 可以换成图片、文件夹、视频
    results = model.predict(
        source=r"F:\yolo\data\images\test",   # 先测测试集图片
        imgsz=640,
        conf=0.25,
        save=True,
        project=r"F:\yolo\runs",
        name="hardhat_predict"
    )

    print("预测完成。")

if __name__ == "__main__":
    main()