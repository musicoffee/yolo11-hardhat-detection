import os
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "hardhat_yolo11n2", "weights", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "demo_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("当前加载模型路径：", MODEL_PATH)

# =========================
# 页面配置
# =========================
st.set_page_config(page_title="安全帽检测系统", layout="wide")
st.title("基于 YOLO11 的安全帽检测系统")
st.caption("支持视频上传、目标检测、结果可视化与检测后视频导出")

# =========================
# 工具函数
# =========================
def box_iou(box1, box2):
    """
    box = [x1, y1, x2, y2]
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


def head_has_helmet(head_box, helmet_boxes, iou_thresh=0.08):
    """
    根据 head 与 helmet 的空间关系判断是否佩戴安全帽
    head_box: [x1, y1, x2, y2]
    helmet_boxes: [[x1, y1, x2, y2], ...]
    """
    for hb in helmet_boxes:
        # IoU 判断
        if box_iou(head_box, hb) >= iou_thresh:
            return True

        # helmet 中心点是否落在 head 框内
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
    conf=0.35,
    iou=0.30,
    min_helmet_area=350,
    min_head_area=500
):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 写出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = 0
    total_heads = 0
    total_helmets = 0
    total_no_helmet = 0

    last_frame_heads = 0
    last_frame_helmets = 0
    last_frame_no_helmet = 0

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

                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h

                class_name = model.names[cls_id].lower()

                # 过滤小框，减少远景密集场景误检
                if class_name == "helmet":
                    if box_area < min_helmet_area:
                        continue
                    helmet_boxes.append([x1, y1, x2, y2, score])

                elif class_name == "head":
                    if box_area < min_head_area:
                        continue
                    head_boxes.append([x1, y1, x2, y2, score])

        total_heads += len(head_boxes)
        total_helmets += len(helmet_boxes)

        helmet_xy = [b[:4] for b in helmet_boxes]

        # 当前帧统计
        frame_no_helmet = 0

        # 先画 helmet
        for x1, y1, x2, y2, score in helmet_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"helmet {score:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # 再画 head / no_helmet
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
                0.55,
                color,
                2
            )

        total_no_helmet += frame_no_helmet

        last_frame_heads = len(head_boxes)
        last_frame_helmets = len(helmet_boxes)
        last_frame_no_helmet = frame_no_helmet

        # 左上角显示当前帧统计
        cv2.putText(frame, f"Current Heads: {last_frame_heads}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2)
        cv2.putText(frame, f"Current Helmets: {last_frame_helmets}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Current No Helmet: {last_frame_no_helmet}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", caption="检测中...")

        progress_bar.progress(min(total_frames / total_frame_count, 1.0))

    cap.release()
    writer.release()

    return {
        "total_frames": total_frames,
        "total_heads": total_heads,
        "total_helmets": total_helmets,
        "total_no_helmet": total_no_helmet,
        "last_frame_heads": last_frame_heads,
        "last_frame_helmets": last_frame_helmets,
        "last_frame_no_helmet": last_frame_no_helmet
    }


# =========================
# 左侧参数区
# =========================
st.sidebar.header("参数设置")

conf_threshold = st.sidebar.slider("置信度阈值", 0.05, 0.95, 0.35, 0.05)
iou_threshold = st.sidebar.slider("NMS IOU 阈值", 0.10, 0.90, 0.30, 0.05)

min_helmet_area = st.sidebar.slider("helmet 最小框面积", 100, 3000, 350, 50)
min_head_area = st.sidebar.slider("head 最小框面积", 100, 4000, 500, 50)

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
                    iou=iou_threshold,
                    min_helmet_area=min_helmet_area,
                    min_head_area=min_head_area
                )

            st.success("检测完成！")

            st.subheader("最终一帧统计")
            col1, col2, col3 = st.columns(3)
            col1.metric("当前帧 head 数量", stats["last_frame_heads"])
            col2.metric("当前帧 helmet 数量", stats["last_frame_helmets"])
            col3.metric("当前帧 no_helmet 数量", stats["last_frame_no_helmet"])

            st.subheader("全视频累计统计")
            col4, col5, col6, col7 = st.columns(4)
            col4.metric("总帧数", stats["total_frames"])
            col5.metric("累计 head 检测次数", stats["total_heads"])
            col6.metric("累计 helmet 检测次数", stats["total_helmets"])
            col7.metric("累计 no_helmet 判定次数", stats["total_no_helmet"])

            st.caption("说明：以上累计统计为视频逐帧检测累计次数，并非唯一目标人数统计。")

            st.subheader("检测后视频")
            st.video(output_video_path)

            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="下载检测结果视频",
                    data=f,
                    file_name="result_video.mp4",
                    mime="video/mp4"
                )
else:
    st.info("请先上传一个视频文件。")