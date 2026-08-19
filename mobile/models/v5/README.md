# Models v5 — drone_yolov8n_p2_closeup

## Training config

| Field | Value |
|-------|-------|
| Model | YOLOv8n-P2 |
| Backbone | pretrained on COCO |
| Dataset | drone_yolo_v5 (close-up drone images) |
| Task | detection (39 classes, citrus diseases + pests + beneficial insects) |
| Epochs | 104 |
| Training time | ~7.3 h (Kaggle GPU T4) |
| Notebook | P2-head training notebook v5 |

## Results

| Metric | Best | Best epoch | Last (epoch 104) |
|--------|------|-----------|-----------------|
| mAP50 | 0.500 | 104 | 0.500 |
| mAP50-95 | 0.313 | 84 | 0.313 |
| Precision | 0.695 | 18 | 0.544 |
| Recall | 0.490 | 100 | 0.491 |

## Files

| File | Description |
|------|-------------|
| `best.onnx` | Best model exported to ONNX (13 MB) — **use this for K230** |
| `best.zip` | Best PyTorch checkpoint (6.2 MB) |
| `last.zip` | Final epoch checkpoint (6.2 MB) |

## Usage

```python
from ultralytics import YOLO

model = YOLO("models/v5/best.zip")
results = model("image.jpg")
```

## Notes

- mAP50 ≈ 0.50 is reasonable for a small n-P2 model trained from scratch on limited drone data.
- Precision starts high (0.70) then drops — the model becomes more conservative over epochs.
- Recall ~0.49 means the model finds about half the objects; this is expected with 640x640 input on small leaf targets at drone altitude.
