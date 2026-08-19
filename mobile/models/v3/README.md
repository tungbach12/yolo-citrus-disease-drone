# YOLOv8n — Version 3 (v3)

- **Trạng thái:** bản train dài của v2 — đã có v4 nhắm riêng drone
- **Ngày:** 2026-08-12 (08:27)
- **Ultralytics:** 8.4.118
- **Cách train:** Kaggle (GPU T4), dataset Roboflow `Citrus-Disease-Detection-1` (39 class),
  `imgsz=640`, `batch=16`, `epochs=192` (cap `time=8` giờ, `patience=15` → có thể dừng sớm hơn 192).

## Khác gì v2

v3 **không đổi augmentation** — chỉ train lâu hơn với batch nhỏ hơn:

| Tham số | v2 | v3 |
|---|---|---|
| epochs | 100 | **192** |
| batch | 32 | **16** |
| patience | 20 | **15** |
| time cap | — | **8h** |
| scale / fliplr / mosaic | 0.5 / 0.5 / 1.0 | giống |
| mixup / copy_paste | 0.0 / 0.0 | giống |

Batch nhỏ + epochs cao = nhiều gradient update hơn → thường nhích mAP lên chút,
nhưng **không** giúp model thích nghi với lá nhỏ nhìn từ drone (đó là việc của v4).

## Metrics

Chưa ghi lại — run này không lưu `results.csv` về repo, và dataset ảnh chưa có local
nên chưa đo lại được. Muốn đo: tải dataset về `data/citrus/` rồi chạy

```bash
venv/Scripts/python.exe -c "from ultralytics import YOLO; print(YOLO('models/v3/best.pt').val(data='data/citrus/data.yaml', imgsz=640).results_dict)"
```

Tham chiếu: v1 mAP50 = 0.531, v2 mAP50 = 0.554.

## Files

| File | Dung lượng | Dùng cho |
|---|---|---|
| `best.pt` | 6.3 MB | Chạy realtime laptop (`MODEL_PATH=../models/v3/best.pt`) |
| `best.onnx` | 12.3 MB | Trung gian, convert ra kmodel |
| `best_ncnn_model/` | 12.1 MB | NCNN — đã nhúng vào APK Android (`v3.ncnn.*`) |

Chưa có `best.kmodel` (chưa convert cho K230).

## Ghi chú

- Cùng dataset, cùng `yolov8n.pt`, cùng 39 class như v1/v2/v4 → so sánh trực tiếp được.
- Trên APK chọn spinner **v3** để đối chiếu với v2/v4 trên cùng góc camera.
- Vì augmentation giống v2, v3 dự kiến cho kết quả **gần giống v2**; chênh lệch rõ rệt
  hơn nằm ở [v4](../v4/README.md).
