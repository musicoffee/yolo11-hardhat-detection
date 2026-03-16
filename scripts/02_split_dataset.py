import os
import random
import shutil

# ===== 原始路径 =====
IMAGE_DIR = r"F:\yolo\data\images"
LABEL_DIR = r"F:\yolo\data\labels_all"

# ===== 输出根目录 =====
OUT_ROOT = r"F:\yolo\data"

# ===== 划分比例 =====
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

def ensure_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

def main():
    ensure_dirs()

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"找到图片数量: {len(image_files)}")

    # 只保留有对应标签的图片
    valid_images = []
    for img_name in image_files:
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(LABEL_DIR, txt_name)
        if os.path.exists(txt_path):
            valid_images.append(img_name)

    print(f"有对应标签的图片数量: {len(valid_images)}")

    random.shuffle(valid_images)

    total = len(valid_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = valid_images[:train_end]
    val_files = valid_images[train_end:val_end]
    test_files = valid_images[val_end:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, files in split_map.items():
        for img_name in files:
            txt_name = os.path.splitext(img_name)[0] + ".txt"

            src_img = os.path.join(IMAGE_DIR, img_name)
            src_txt = os.path.join(LABEL_DIR, txt_name)

            dst_img = os.path.join(OUT_ROOT, "images", split, img_name)
            dst_txt = os.path.join(OUT_ROOT, "labels", split, txt_name)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_txt, dst_txt)

    print("数据集划分完成：")
    print(f"train: {len(train_files)}")
    print(f"val:   {len(val_files)}")
    print(f"test:  {len(test_files)}")

if __name__ == "__main__":
    main()