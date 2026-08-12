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

# ====== CẤU HÌNH ======
MODEL_PATH = "../models/best.pt"     # đổi nếu file model nằm chỗ khác
CONF = float(__import__("os").environ.get("CONF", "0.25"))  # ngưỡng tin cậy — hạ xuống 0.15-0.20 để tăng recall
IMGSZ = 640                # phải khớp imgsz lúc export
CAMERA = 0                 # webcam mặc định (0). Đổi 1, 2... nếu nhiều cam
# ======================

# Load model (nếu máy có GPU thì tự dùng, không thì CPU)
model = YOLO(MODEL_PATH)
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Nhấn 'q' để thoát.")
import time
fps_last = time.time()
fps_accum = 0
fps_frames = 0
blank_frames = 0
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

    # Hiển thị
    cv2.imshow("YOLOv8 - Webcam (q de thoat)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã thoát.")