# -*- coding: utf-8 -*-
# Train YOLOv8n từ đầu trên Kaggle với Dataset Cận Cảnh (Close-Up) Roboflow
#
# CÁCH DÙNG TRÊN KAGGLE:
# 1. Bật GPU T4 x2 + Internet ON trong Settings
# 2. Tạo secret ROBOFLOW_API_KEY trong Kaggle Add-ons -> Secrets
# 3. Chạy:
#    !python scripts/train_drone_kaggle_v4.py
#
import os
import sys
import glob
import shutil
import yaml

# ====== CẤU HÌNH TRAIN DATASET CẬN CẢNH ROBOFLOW ======
MODEL_NAME      = "yolov8n.pt"          # Model Nano (3.2M params) siêu nhẹ cho chip Kendryte K230
EPOCHS          = 150
IMGSZ           = 640                   # Độ phân giải chuẩn cho K230 ONNX export
BATCH           = 32                    # Batch size 32 cho yolov8n
PATIENCE        = 20
OUT_DIR         = "/kaggle/working/drone_yolo_v4_close_up"
# ======================================================


def get_roboflow_key():
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
        if key: return key
    except Exception:
        pass
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ["ROBOFLOW_API_KEY"]
    try:
        for line in open(".env"):
            line = line.strip()
            if line.startswith("ROBOFLOW_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    raise ValueError("Thiếu ROBOFLOW_API_KEY. Thêm secret tên ROBOFLOW_API_KEY trên Kaggle!")


def download_dataset():
    api_key = get_roboflow_key()
    print("Đã lấy ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("trantungbach26-gmail-com").project("citrus-disease-detection-yoydc-ahtka")
    project.version(1).download("yolov8")
    print("Dataset cận cảnh đã tải về /kaggle/working/")

    candidates = glob.glob("/kaggle/working/*/data.yaml")
    dataset_path = os.path.dirname(candidates[0]) if candidates else "/kaggle/working/citrus-disease-detection-1"
    return dataset_path


def train():
    from ultralytics import YOLO
    from ultralytics.utils import callbacks

    os.makedirs(OUT_DIR, exist_ok=True)
    dataset_path = download_dataset()
    data_yaml = os.path.join(dataset_path, "data.yaml")

    def _backup(trainer):
        try:
            src = os.path.join(trainer.save_dir, "weights", "best.pt")
            shutil.copy(src, os.path.join(OUT_DIR, "best_checkpoint.pt"))
            print(f"  [backup epoch {trainer.epoch}] -> {OUT_DIR}/best_checkpoint.pt", flush=True)
        except Exception:
            pass

    callbacks.default_callbacks["on_fit_epoch_end"].append(_backup)

    print(f"Train từ đầu {MODEL_NAME}: epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}")
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=0,
        seed=42,
        time=8,              # Tối đa 8 giờ session Kaggle
        cache=True,
        workers=2,
        # Augmentations biến đổi scale & góc chụp cho lá cận cảnh
        scale=0.8,           # Thu nhỏ/phóng to ngẫu nhiên lá từ 20% đến 180%
        fliplr=0.5,          # Lật ngang
        mosaic=1.0,          # Mosaic 4 ảnh cận cảnh ghép lại (giúp model nhìn nhiều góc)
        mixup=0.15,          # Trộn ảnh tạo nhiễu ánh sáng
        copy_paste=0.2,      # Trộn vết bệnh
        cos_lr=True,         # Giảm learning rate theo CosineAnnealing
        project="/kaggle/working/runs",
        name="drone_yolov8n_closeup",
    )

    metrics = model.val()
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    # Export ONNX dành cho chip K230 Drone
    results_dir = "/kaggle/working/runs/drone_yolov8n_closeup/weights"
    best_path = os.path.join(results_dir, "best.pt")
    if os.path.exists(best_path):
        best_m = YOLO(best_path)
        best_m.export(format="onnx", imgsz=IMGSZ, opset=11, simplify=True)
        shutil.copy(best_path, os.path.join(OUT_DIR, "best.pt"))
        onnx_src = os.path.join(results_dir, "best.onnx")
        if os.path.exists(onnx_src):
            shutil.copy(onnx_src, os.path.join(OUT_DIR, "best.onnx"))
        print(f"Đã copy best.pt và best.onnx vào {OUT_DIR}")

    print("\n>>> TẢI KẾT QUẢ: Panel bên phải tab 'Output' -> biểu tượng Download all.")


if __name__ == "__main__":
    train()
