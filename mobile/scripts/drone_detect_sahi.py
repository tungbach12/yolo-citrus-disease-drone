# -*- coding: utf-8 -*-
# Drone detection bằng SAHI (Slicing Aided Hyper Inference)
#
# Giải quyết vấn đề: ảnh train chụp gần, drone chụp xa 3-4m -> đốm bệnh nhỏ.
# SAHI cắt ảnh thành các tile 640x640 (có overlap), chạy YOLO trên từng tile,
# rồi gộp + NMS. Mỗi tile = "zoom vào" -> đốm nhỏ thành lớn -> recall cao hơn.
#
# CÁCH DÙNG:
#   python drone_detect_sahi.py <ảnh hoặc thư mục hoặc video>
#   python drone_detect_sahi.py                    # mở webcam
#   CONF=0.15 python drone_detect_sahi.py image.jpg
#
# Yêu cầu: pip install sahi ultralytics opencv-python

import os
import sys

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ====== CẤU HÌNH ======
MODEL_PATH = os.environ.get("MODEL_PATH", "../models/v2/best.pt")
CONF = float(os.environ.get("CONF", "0.15"))   # thấp để tăng recall
SLICE = int(os.environ.get("SLICE", "640"))    # kích thước tile
OVERLAP = float(os.environ.get("OVERLAP", "0.25"))  # overlap giữa các tile
# ======================


def main():
    import cv2

    source = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Model: {MODEL_PATH} | conf={CONF} | slice={SLICE} | overlap={OVERLAP}")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=CONF,
        device="cuda:0",
    )
    print(f"Loaded. Classes: {len(model.class_name) if hasattr(model, 'class_name') else '?' }")

    if source is None:
        _run_webcam(model)
        return

    if os.path.isdir(source):
        _run_folder(model, source)
    elif source.lower().endswith((".mp4", ".avi", ".mov")):
        _run_video(model, source)
    else:
        _run_image(model, source)


def _predict_sahi(model, image_bgr):
    """Chạy SAHI trên 1 ảnh BGR (OpenCV). Trả về ảnh đã annotate."""
    import cv2
    import numpy as np

    # SAHI cần RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = get_sliced_prediction(
        image_rgb,
        model,
        slice_height=SLICE,
        slice_width=SLICE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )

    # Vẽ box lên ảnh gốc
    annotated = image_bgr.copy()
    n = len(result.object_prediction_list)
    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = pred.category.name
        score = pred.score.value

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(annotated, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(annotated, f"Detections: {n}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return annotated, n


def _run_image(model, path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        print(f"Không đọc được: {path}")
        return
    print(f"Ảnh: {path} ({img.shape[1]}x{img.shape[0]})")
    annotated, n = _predict_sahi(model, img)
    print(f"  -> {n} detections")

    out = os.path.join(os.path.dirname(path), "sahi_" + os.path.basename(path))
    cv2.imwrite(out, annotated)
    print(f"  Đã lưu: {out}")

    if not os.environ.get("HEADLESS"):
        cv2.imshow("SAHI Detection (q de thoat)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_folder(model, folder):
    import glob
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(folder, e)))
    if not imgs:
        print(f"Không có ảnh trong: {folder}")
        return
    print(f"Tìm thấy {len(imgs)} ảnh")
    for p in sorted(imgs):
        _run_image(model, p)


def _run_video(model, path):
    import cv2
    import time
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Không mở được video: {path}")
        return
    print(f"Video: {path} — nhấn 'q' để thoát")
    fps_last = time.time()
    fps_frames = 0
    fps_val = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, n = _predict_sahi(model, frame)
        fps_frames += 1
        now = time.time()
        if now - fps_last >= 1.0:
            fps_val = fps_frames / (now - fps_last)
            fps_frames = 0
            fps_last = now
        cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SAHI Detection (q de thoat)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def _run_webcam(model):
    import cv2
    import time
    print("Webcam — nhấn 'q' để thoát")
    cap = None
    for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
        cap = cv2.VideoCapture(0, api) if api else cv2.VideoCapture(0)
        if cap.isOpened():
            break
        cap.release()
    if cap is None or not cap.isOpened():
        print("Không mở được webcam")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_last = time.time()
    fps_frames = 0
    fps_val = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        annotated, n = _predict_sahi(model, frame)
        fps_frames += 1
        now = time.time()
        if now - fps_last >= 1.0:
            fps_val = fps_frames / (now - fps_last)
            fps_frames = 0
            fps_last = now
        cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SAHI Detection (q de thoat)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
