# YOLOv8n — Version 2 (v2)

- **Trạng thái:** hiện tại — dùng cho laptop + K230
- **Ngày:** 2026-08-11 (train 100 epochs, best at epoch 70)
- **Epochs:** 100 (early-stop ~epoch 85)
- **Cách train:** Kaggle resume từ v1, imgsz 640, batch 16, cos_lr. Vẫn chưa dùng augment tăng recall mới (class_weights/fliplr/scale) — cấu hình đó sẵn trong notebook cho lần train tới.

## Metrics (val, best at epoch 70)

| Metric | Giá trị |
|---|---|
| precision | 0.6628 |
| recall | 0.4528 |
| mAP50 | 0.5541 |
| mAP50-95 | 0.3662 |

So với v1: mAP50 tăng 0.531 → 0.554 (+2.3 điểm), recall/P cải thiện nhẹ.

## Files

| File | Dung lượng | Dùng cho |
|---|---|---|
| `best.pt` | 6.3 MB | Chạy realtime trên laptop (`scripts/realtime_cam.py`) |
| `last.pt` | 6.3 MB | Checkpoint cuối run — dùng để `resume=True` nếu train tiếp |
| `best.onnx` | 12.3 MB | Trung gian, convert ra kmodel |
| `best.kmodel` | 12.2 MB | Chạy trên CanMV K230 (`scripts/k230_yolov8_det.py`) |

## Ghi chú

- Vẫn P cao / R thấp: model "nhát tay", bỏ sót nhiều bệnh. Hạ `conf` khi chạy
  (`CONF=0.15` cho webcam / `conf_threshold` K230) để tăng recall thực tế.
- Lần train tiếp theo (nếu cần) dùng notebook `train_yolov8n_drone_kaggle_resume.ipynb`
  với `class_weights=True, fliplr=0.5, scale=0.5` để kéo recall lên ~0.7.