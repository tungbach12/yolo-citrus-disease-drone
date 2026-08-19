# YOLOv8n — Version 4 (v4)

- **Trạng thái:** hiện tại — bản nhắm riêng cho drone (3-4 m), dùng trong APK Android
- **Ngày:** 2026-08-12 (19:16)
- **Ultralytics:** 8.4.118
- **Cách train:** Kaggle (GPU T4), dataset Roboflow `Citrus-Disease-Detection-1` (39 class),
  `imgsz=640`, `batch=32`, `epochs=192` (cap `time=8` giờ, `patience=20` → có thể dừng sớm),
  notebook `notebooks/train_drone_kaggle_v4.ipynb`.

## Khác gì v3 — 3 augment mới cho drone

| Tham số | v3 | **v4** | Vì sao |
|---|---|---|---|
| `scale` | 0.5 | **0.8** | Lá co giãn 20%–180% (v3 chỉ 50%–150%). Dataset toàn ảnh cận cảnh — scale 0.8 giả lập **lá nhỏ như nhìn từ drone 3-4 m**. Đây là thay đổi quan trọng nhất |
| `mixup` | 0.0 | **0.15** | Trộn 2 ảnh → chịu nhiễu ánh sáng ngoài trời tốt hơn |
| `copy_paste` | 0.0 | **0.2** | Dán vết bệnh sang ảnh khác → thấy nhiều biến thể vết bệnh |
| epochs / batch / patience | 192 / 16 / 15 | 192 / 32 / 20 | gần giống nhau |

Vì augmentation mạnh hơn (nhất là scale 0.8), v4 kỳ vọng **recall cao hơn hẳn khi lá nhỏ**
— đúng bài toán drone quét vườn. Cái giá: precision có thể tụt nhẹ (false positive nhiều hơn).

## Metrics

Chưa ghi lại — không có `results.csv` của run này, dataset ảnh chưa có local.
Muốn đo: tải dataset về `data/citrus/` rồi chạy

```bash
venv/Scripts/python.exe -c "from ultralytics import YOLO; print(YOLO('models/v4/best.pt').val(data='data/citrus/data.yaml', imgsz=640).results_dict)"
```

Tham chiếu: v1 mAP50 = 0.531, v2 mAP50 = 0.554.

## Files

| File | Dung lượng | Dung cho |
|---|---|---|
| `best.pt` | 6.3 MB | Chạy realtime laptop (`MODEL_PATH=../models/v4/best.pt`) |
| `best_ncnn_model/` | 12.1 MB | NCNN — đã nhúng vào APK Android (`v4.ncnn.*`) |

Chưa có `best.onnx` (chưa cần), chưa có `best.kmodel` (chưa convert cho K230).
Nếu muốn dùng v4 trên K230: export ONNX opset 11 → convert kmodel như [v1](../v1/README.md).

## Ghi chú

- Cùng dataset, cùng `yolov8n.pt`, cùng 39 class như v1/v2/v3 → so sánh trực tiếp được.
- Trên APK: chọn spinner **v4** + **Bật SAHI** + tile **320** là cấu hình mạnh nhất cho
  lá bệnh nhỏ từ drone xa (đổi lấy FPS thấp ~1-2).
- So sánh trực tiếp: cùng góc camera, switch v1→v2→v3→v4, ghi nhận số bbox + score.
