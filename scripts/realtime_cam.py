# -*- coding: utf-8 -*-
# Chạy YOLOv8 realtime bằng webcam trên laptop (dùng best.pt sau khi train).
#
# CÁCH DÙNG:
#   1) Đặt file best.pt cùng thư mục với script này (hoặc sửa MODEL_PATH dưới đây)
#   2) python realtime_cam.py
#   3) Nhấn 'q' để thoát.
#
# Yêu cầu đã cài: pip install ultralytics opencv-python
# Lưu ý: laptop hiện dùng torch bản CPU -> chạy chậm hơn, nhưng yolov8n vẫn test được.

from ultralytics import YOLO
import cv2

import os

# ====== CẤU HÌNH ======
MODEL_PATH = os.environ.get("MODEL_PATH", "../models/v2/best.pt")  # mặc định v2; set MODEL_PATH=../models/v4/best.pt để test v4
CONF = float(os.environ.get("CONF", "0.25"))  # ngưỡng tin cậy — hạ xuống 0.15-0.20 để tăng recall
IMGSZ = int(os.environ.get("IMGSZ", "640"))   # phải khớp imgsz lúc export
DEVICE = os.environ.get("DEVICE", "cuda:0")    # "cuda:0"=GPU, "cpu"=CPU. Mặc định GPU nếu có.
CAMERA = 0                 # webcam mặc định (0). Đổi 1, 2... nếu nhiều cam
# ======================

# Load model (nếu máy có GPU thì tự dùng, không thì CPU)
model = YOLO(MODEL_PATH)
# Ép sang GPU nếu DEVICE yêu cầu (ultralytics không auto-detect trong mọi trường hợp)
if DEVICE != "cpu":
    try:
        import torch
        if torch.cuda.is_available():
            model.to(DEVICE)
    except Exception as e:
        print(f"Không chuyển được GPU ({e}), dùng CPU.", flush=True)
print(f"Đã load: {MODEL_PATH} | conf={CONF} | device={model.device}", flush=True)

# Lấy frame: MSMF mặc định trên Windows hay lỗi -1072873822 (không grab được).
# Dùng DSHOW ổn định hơn; nếu fail thì fallback về mặc định.
cap = None
for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
    if api == 0:
        cap = cv2.VideoCapture(CAMERA)
    else:
        cap = cv2.VideoCapture(CAMERA, api)
    if cap.isOpened():
        print(f"Webcam mở bằng backend: {api}", flush=True)
        break
    cap.release()

if cap is None or not cap.isOpened():
    print("Không mở được webcam. Kiểm tra chỉ số CAMERA hoặc cắm webcam.")
    exit(1)

# Buffer thấp để giảm trễ (mặc định 1 hoặc cao hơn tùy backend)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
]
current_res_idx = 1  # mặc định 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTIONS[current_res_idx][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTIONS[current_res_idx][1])

print(f"Resolution: {RESOLUTIONS[current_res_idx][0]}x{RESOLUTIONS[current_res_idx][1]}", flush=True)
print("Phim: 'r' = doi resolution | 'q' = thoat", flush=True)
import time
fps_last = time.time()
fps_accum = 0
fps_frames = 0
blank_frames = 0
res_notif_timer = 0  # hiển thị thông báo resolution trong 2 giây
while True:
    ret, frame = cap.read()
    if not ret:
        blank_frames += 1
        if blank_frames > 30:
            print("Quá nhiều frame trắng — webcam không trả ảnh được (kiểm tra chỉ số CAMERA).", flush=True)
            break
        continue

    # Chạy dự đoán từng khung hình
    results = model.predict(
        source=frame,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )
    annotated = results[0].plot()   # vẽ khung + nhãn lên ảnh

    # FPS đếm bằng khoảng thời gian thật, cập nhật mỗi ~1s
    fps_frames += 1
    now = time.time()
    if now - fps_last >= 1.0:
        fps_accum = fps_frames / (now - fps_last)
        fps_frames = 0
        fps_last = now
    cv2.putText(annotated, f"FPS: {fps_accum:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Phím tắt: 'r' = đổi resolution, 'q' = thoát
    key = cv2.waitKey(1) & 0xFF
    msg = ""
    if key == ord('r'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        current_res_idx = (current_res_idx + 1) % len(RESOLUTIONS)
        w, h = RESOLUTIONS[current_res_idx]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if actual_w == w and actual_h == h:
            msg = f"Resolution: {int(w)}x{int(h)}"
        else:
            msg = f"Resolution: {int(actual_w)}x{int(actual_h)} (yeu cau {int(w)}x{int(h)} khong duoc)"
        print(msg, flush=True)
        res_notif_timer = time.time() + 2.0
    if key == ord('q'):
        break

    # Hiển thị thông báo resolution (2 giây sau khi đổi)
    if res_notif_timer > time.time():
        color = (0, 255, 0) if "yeu cau" not in msg else (0, 0, 255)
        cv2.rectangle(annotated, (10, 60), (480, 100), (0, 0, 0), -1)
        cv2.putText(annotated, msg, (15, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Hiển thị
    cv2.imshow("YOLOv8 - Webcam (q de thoat)", annotated)

cap.release()
cv2.destroyAllWindows()
print("Đã thoát.")