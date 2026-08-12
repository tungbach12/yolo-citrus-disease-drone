# YOLOv8n — Version 1 (v1)

- **Trạng thái:** bản đầu, đã thay bằng v2
- **Ngày:** 2026-08-11
- **Epochs:** 50
- **Cách train:** Kaggle (GPU T4), dataset Roboflow citrus 16.245 ảnh, imgsz 640, class_weights chưa bật

## Metrics (val)

| Metric | Giá trị |
|---|---|
| precision | 0.6478 |
| recall | 0.4577 |
| mAP50 | 0.5311 |
| mAP50-95 | 0.3424 |

## Files

| File | Dung lượng | Dùng cho |
|---|---|---|
| `best.pt` | 5.7 MB | Chạy realtime trên laptop (`scripts/realtime_cam.py`) |
| `best.onnx` | 12.3 MB | Trung gian, convert ra kmodel |
| `best.kmodel` | 12.2 MB | Chạy trên CanMV K230 (`scripts/k230_yolov8_det.py`) |

## Ghi chú

- Recall thấp (0.46) — model bỏ sót nhiều đốm bệnh, chưa tối ưu cho drone quét vườn.
- Không có `last.pt` (checkpoint cuối) trong bản này.