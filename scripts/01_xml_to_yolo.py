import os
import xml.etree.ElementTree as ET

# ===== 路径 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(BASE_DIR, "data", "annotations")
OUTPUT_LABEL_DIR = os.path.join(BASE_DIR, "data", "labels_all")

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ===== 类别映射 =====
# 只保留 helmet 和 head，忽略 person
CLASS_MAP = {
    "helmet": 0,
    "Helmet": 0,
    "head": 1,
    "Head": 1,
}

def convert_box(img_w, img_h, xmin, ymin, xmax, ymax):
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"XML 缺少 size 节点: {xml_path}")

    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()

        # 只保留 helmet / head
        if cls_name not in CLASS_MAP:
            continue

        cls_id = CLASS_MAP[cls_name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x, y, w, h = convert_box(img_w, img_h, xmin, ymin, xmax, ymax)

        # 防止越界
        x = min(max(x, 0), 1)
        y = min(max(y, 0), 1)
        w = min(max(w, 0), 1)
        h = min(max(h, 0), 1)

        yolo_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return yolo_lines

def main():
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
    print(f"找到 XML 文件数量: {len(xml_files)}")

    converted = 0

    for xml_file in xml_files:
        xml_path = os.path.join(XML_DIR, xml_file)
        txt_name = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_name)

        try:
            lines = parse_xml(xml_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            converted += 1
        except Exception as e:
            print(f"[跳过] {xml_file} 处理失败: {e}")

    print(f"转换完成，共生成 {converted} 个 YOLO 标签文件。")
    print(f"输出目录: {OUTPUT_LABEL_DIR}")

if __name__ == "__main__":
    main()