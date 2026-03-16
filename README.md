# YOLO11 Hard Hat Detection

基于 YOLO11 的安全帽佩戴检测系统，支持：

- 使用 Kaggle 安全帽数据集进行训练
- 将 VOC XML 标注转换为 YOLO 格式
- 使用 YOLO11 进行目标检测训练
- 使用 Streamlit 构建视频检测展示页面
- 基于 `helmet` 和 `head` 的空间关系判断未佩戴安全帽

Environment
pip install -r requirements.txt
Training
python scripts/01_xml_to_yolo.py
python scripts/02_split_dataset.py
python scripts/03_train.py
Predict
python scripts/04_predict.py
Run Web Demo
streamlit run app.py
Dataset

本项目使用 Kaggle 上公开的安全帽检测数据集。
由于数据集版权和体积原因，仓库中不直接提供原始数据，请自行下载后放入对应目录。

Notes

本仓库默认不上传训练权重、原始数据集和训练输出结果

若需复现，请先准备数据集并修改相关路径
## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── hardhat.yaml
├── scripts/
│   ├── 01_xml_to_yolo.py
│   ├── 02_split_dataset.py
│   ├── 03_train.py
│   └── 04_predict.py
