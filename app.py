import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO

# =========================
# 基础配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "hardhat_yolo11n", "weights", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "demo_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="安全帽检测系统", layout="wide")
st.title("基于 YOLO11 的安全帽佩戴检测系统")

# =========================
# 工具函数
# =========================
def box_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def head_has_helmet(head_box, helmet_boxes, iou_thresh=0.10):
    """
    判断 head 附近是否有 helmet
    """
    for hb in helmet_boxes:
        if box_iou(head_box, hb) >= iou_thresh:
            return True

        # 再加一个中心点包含的简单规则，增强鲁棒性
        hx1, hy1, hx2, hy2 = hb
        cx = (hx1 + hx2) / 2
        cy = (hy1 + hy2) / 2

        if head_box[0] <= cx <= head_box[2] and head_box[1] <= cy <= head_box[3]:
            return True

    return False


def process_video(
    input_video_path,
    output_video_path,
    model,
    conf=0.25,
    iou=0.45
):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = 0
    total_heads = 0
    total_helmets = 0
    total_no_helmet = 0

    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frame_count <= 0:
        total_frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        result = results[0]

        boxes = result.boxes
        helmet_boxes = []
        head_boxes = []

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                score = float(b.conf[0].item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                class_name = model.names[cls_id].lower()

                if class_name == "helmet":
                    helmet_boxes.append([x1, y1, x2, y2, score])
                elif class_name == "head":
                    head_boxes.append([x1, y1, x2, y2, score])

        total_heads += len(head_boxes)
        total_helmets += len(helmet_boxes)

        helmet_xy = [b[:4] for b in helmet_boxes]

        # 先画 helmet
        for x1, y1, x2, y2, score in helmet_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"helmet {score:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 再画 head，并判断 no_helmet
        frame_no_helmet = 0
        for x1, y1, x2, y2, score in head_boxes:
            has_helmet = head_has_helmet([x1, y1, x2, y2], helmet_xy, iou_thresh=0.08)

            if has_helmet:
                color = (255, 180, 0)
                label = f"head {score:.2f}"
            else:
                color = (0, 0, 255)
                label = f"no_helmet {score:.2f}"
                frame_no_helmet += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        total_no_helmet += frame_no_helmet

        # 左上角统计信息
        cv2.putText(frame, f"Heads: {len(head_boxes)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2)
        cv2.putText(frame, f"Helmets: {len(helmet_boxes)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"No Helmet: {frame_no_helmet}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)

        # 页面实时显示一帧
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", caption="检测中...")

        progress_bar.progress(min(total_frames / total_frame_count, 1.0))

    cap.release()
    writer.release()

    return {
        "total_frames": total_frames,
        "total_heads": total_heads,
        "total_helmets": total_helmets,
        "total_no_helmet": total_no_helmet
    }


# =========================
# 页面侧边栏
# =========================
st.sidebar.header("参数设置")
conf_threshold = st.sidebar.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider("NMS IOU 阈值", 0.10, 0.90, 0.45, 0.05)

uploaded_file = st.file_uploader("上传一个视频文件", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.subheader("原视频")
    st.video(uploaded_file)

    if st.button("开始检测"):
        if not os.path.exists(MODEL_PATH):
            st.error(f"模型不存在：{MODEL_PATH}")
        else:
            with st.spinner("正在加载模型并处理视频，请稍等..."):
                model = YOLO(MODEL_PATH)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(uploaded_file.read())
                    input_video_path = tmp_in.name

                output_video_path = os.path.join(OUTPUT_DIR, "result_video.mp4")

                stats = process_video(
                    input_video_path=input_video_path,
                    output_video_path=output_video_path,
                    model=model,
                    conf=conf_threshold,
                    iou=iou_threshold
                )

            st.success("检测完成！")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总帧数", stats["total_frames"])
            col2.metric("检测到 head 总数", stats["total_heads"])
            col3.metric("检测到 helmet 总数", stats["total_helmets"])
            col4.metric("未佩戴安全帽累计次数", stats["total_no_helmet"])

            st.subheader("检测后视频")
            with open(output_video_path, "rb") as f:
                st.video(f.read())

            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="下载检测结果视频",
                    data=f,
                    file_name="result_video.mp4",
                    mime="video/mp4"
                )
else:
    st.info("请先上传一个视频文件。")