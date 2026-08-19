# 05 — Project Walkthrough (Bản đồ tổng thể)

> Tài liệu tổng hợp: mục tiêu project, pipeline end-to-end, cấu trúc thư mục, cách chạy từng phần, và bài học từ quá trình làm. Đọc sau khi đã xem 01-04.

## 1. Mục tiêu project

**Drone nông nghiệp phát hiện bệnh cam realtime.** Drone (bo K230) bay trên ruộng cam → camera phát hiện lá/bệnh/pest → hiển thị box + tên bệnh lên màn → người nông dân biết vùng nào cần xử lý.

**Yêu cầu kỹ thuật:**
- Phát hiện **39 class** (bệnh + pest + côn trùng có ích + lá khỏe).
- **Realtime** trên K230 (~10-20 FPS).
- Test trên **laptop webcam** trước khi deploy drone (dùng `best.pt`).

---

## 2. Pipeline end-to-end

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATASET (Roboflow)                                        │
│    Citrus Disease Detection v1 — 39 class, YOLOv8 format     │
│    download trực tiếp trong notebook                         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAIN (Kaggle GPU T4)                                     │
│    notebooks/train_yolov8n_drone_kaggle.ipynb               │
│    YOLOv8n, 50 epochs, patience 15, auto-backup mỗi epoch   │
│    → best.pt (5.4 MB)                                        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EXPORT ONNX (laptop CPU, 25s)                            │
│    ultralytics: YOLO(best.pt).export(format=onnx,...)       │
│    → best.onnx (11.7 MB)                                     │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. COMPILE KMODEL (laptop, nncase 2.10.0 + .NET 7)          │
│    scripts/convert_to_kmodel.py                             │
│    → best.kmodel (11.6 MB)                                   │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌──────────────┬───────────────────────────────────────────────┘
│              │
↓ 5a. LAPTOP    ↓ 5b. K230 DRONE
│              │
│ scripts/     │ /sdcard/best.kmodel
│ realtime_    │ /sdcard/k230_yolov8_det.py
│ cam.py       │ → CanMV IDE → Run
│ → webcam     │ → camera realtime
│   realtime   │ → màn LCD 800×480
└──────────────┘
```

---

## 3. Cấu trúc thư mục

```
E:\Dev\Projects\Yolo\
├── models/
│   ├── best.pt              # weights train (laptop webcam + nguồn export ONNX)
│   ├── best.onnx            # ONNX trung gian (nguồn compile kmodel)
│   ├── best.kmodel          # KPU binary deploy K230
│   └── yolov8n.pt           # pretrained COCO (nguồn transfer learning)
├── notebooks/
│   ├── train_yolov8n_drone_kaggle.ipynb   # train trên Kaggle (đã fix)
│   └── train_yolov8n_drone.ipynb          # train trên Colab (variant)
├── scripts/
│   ├── convert_to_kmodel.py   # ONNX → kmodel (nncase 2.10.0)
│   ├── realtime_cam.py        # test webcam laptop
│   ├── k230_yolov8_det.py     # chạy trên K230
│   └── rebuild_pt.py          # rebuild .pt từ Kaggle web-UI zip (reference)
├── data/
│   └── citrus/                # metadata dataset (không có ảnh)
│       ├── data.yaml          # 39 class names + cấu hình
│       ├── README.dataset.txt
│       └── README.roboflow.txt
└── docs/
    ├── 01_yolo_architecture.md   ← bạn đang ở đây (bản đồ)
    ├── 02_training_process.md
    ├── 03_data_prep.md
    ├── 04_export_deploy.md
    └── 05_project_walkthrough.md
```

---

## 4. Cách chạy từng phần

### 4.1 Train (trên Kaggle)

1. Tải `notebooks/train_yolov8n_drone_kaggle.ipynb` lên Kaggle.
2. Settings: Accelerator = **GPU T4 x2**, Internet = **ON**.
3. Chạy lần lượt cell 0 → cell 14.
4. Sau khi xong, download `best.zip` (hoặc `best_checkpoint.pt`) từ tab **Output**.

> Xem [02_training_process.md](02_training_process.md) để hiểu train loop, loss, metrics.

### 4.2 Export ONNX (trên laptop)

```powershell
$env:PATH = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
python -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', imgsz=640, opset=11, simplify=True)"
```

Output: `models/best.onnx` (11.7 MB). Mất ~25s trên CPU.

> Xem [04_export_deploy.md](04_export_deploy.md) mục 2.

### 4.3 Compile kmodel (trên laptop)

```powershell
python scripts/convert_to_kmodel.py
```

Yêu cầu: `nncase==2.10.0`, `nncase_kpu==2.10.0`, `.NET 7.0 x64` đã cài.

Output: `models/best.kmodel` (11.6 MB).

> Xem [04_export_deploy.md](04_export_deploy.md) mục 3.

### 4.4 Test laptop webcam

```powershell
python scripts/realtime_cam.py
```

Webcam mở, vẽ box + label realtime. Nhấn `q` thoát.

> Yêu cầu: `pip install ultralytics opencv-python`. Laptop CPU → ~5-15 FPS (yolov8n nhẹ).

### 4.5 Deploy K230

1. Copy `models/best.kmodel` → thẻ SD `/sdcard/best.kmodel`.
2. Copy `scripts/k230_yolov8_det.py` → thẻ SD `/sdcard/`.
3. Cắm thẻ vào K230, mở CanMV IDE.
4. Mở `k230_yolov8_det.py`, Connect (icon 1), Run (icon 2).
5. Camera + màn LCD chạy realtime.

> Xem [04_export_deploy.md](04_export_deploy.md) mục 4-5.

---

## 5. Bài học thực tế — troubleshooting

### 5.1 Kaggle kernel ERROR — không download được

**Tình huống:** Kernel `natsumeeei/capstion-v1` rơi vào **ERROR** state sau khi cell-11 (preview) crash với `FileNotFoundError`. `kernels_list_files` trả 0 file → `kaggle kernels output` (dù có `--file-pattern`) treo vô hạn.

**Root cause:** Cell preview hardcode path `predict/predict/...jpg` — đường dẫn không tồn tại (ultralytics tạo `predict/`, `predict2/`, ... tùy session) → `display(Image(filename=...))` ném FileNotFoundError → notebook crash trước khi cell-13 (export ONNX) chạy.

**Fix:**
- Cell-11 wrap try/except + glob tự tìm ảnh thực tế (đã fix trong notebook hiện tại).
- **Recovery thay vì re-train:** Download `best.zip` qua **Kaggle web UI** (tab Output → biểu tượng download) — web UI vẫn cho download dù API chặn.

**Bài học:**
1. Kaggle API `kernels_list_files` **phụ thuộc status kernel** — ERROR state = 0 file. Đừng cố API, dùng web UI.
2. Cell preview **phải fail-safe** — không bao giờ để 1 cell display crash chặn cell export sau nó.
3. **Auto-backup callback** cứu project: `best_checkpoint.pt` được copy ra Output folder mỗi epoch → kernel crash vẫn download được bản train đến lúc đó.

### 5.2 Web UI download unzip .pt — rebuild thủ công

**Tình huống:** `best.zip` tải từ Kaggle web UI **không phải** file `.pt` nén nguyên vẹn — mà là **nội dung unzip** của `.pt` (PyTorch .pt bản chất là zip): `data.pkl`, `.data/`, `data/`, `version`, `byteorder`, ...

**Vấn đề:** Re-zip flat (không prefix folder) → PyTorch báo `file in archive is not in a subdirectory: .format_version`.

**Fix:** [scripts/rebuild_pt.py](../../scripts/rebuild_pt.py) — re-zip với **prefix folder** khớp tên file (`best/` cho `best.pt`). PyTorch expect entries dạng `t/data.pkl`, không phải `data.pkl` flat.

**Bài học:** Khi công cụ (Kaggle web UI) "hơi khác" spec, hiểu format file (.pt = zip có cấu trúc) → tự fix được thay vì chờ tool.

### 5.3 nncase 2.10.0 API khác tutorial cũ

**Tình huống:** Script convert ban đầu (viết theo tutorial cũ) fail:
```
TypeError: Compiler.__init__() missing 1 required positional argument: 'compile_options'
```

**Root cause:** nncase 2.10.0 đổi API:
- `Compiler()` (không arg) → `Compiler(compile_options)`.
- `import_onnx(path, input_layout)` → `import_onnx(model_bytes, ImportOptions)`.
- `compile(options)` → `compile()` (options ở constructor).
- `RuntimeTensorLayout("NCHW")` bị bỏ.

**Fix:** Rewrite script theo API mới (đã làm). Test bằng `inspect.signature()` để xem signature thực.

**Bài học:** Tutorial/docs version cũ ≠ API hiện tại. Khi gặp TypeError signature, dùng `python -c "import inspect; print(inspect.signature(Class.__init__))"` xem signature thật, không tin docs.

### 5.4 nncase cần .NET 7

**Tình huống:** `import nncase` fail: `Failed to initialize hostfxr: 0x80008096`.

**Root cause:** nncase 2.10 build trên .NET 7. Máy có .NET 9 nhưng **không có 7** → runtime không tương thích ngược.

**Fix:** Cài `.NET 7.0 x64 Desktop Runtime` (download từ Microsoft).

**Bài học:** Dependency runtime phiên bản cụ thể — đọc error message kỹ, không đoán.

---

## 6. Tham số đã chốt

| Tham số | Giá trị | Lý do |
|---|---|---|
| Model | YOLOv8n | K230 giới hạn, cần realtime |
| Epochs | 50 | yolov8n hội tụ nhanh, Kaggle không bị cut |
| Patience | 15 | Dừng sớm nếu val không cải thiện |
| Batch | 16 | T4 VRAM 15G sweet spot |
| Imgsz | 640 | Khớp train + export + K230 |
| Optimizer | SGD momentum (mặc định) | Generalize tốt cho detection |
| LR schedule | Cosine (`cos_lr=True`) | Mượt, hội tụ tốt |
| Augmentation | mosaic + fliplr 0.5, flipud 0 | Drone top-down, không lật dọc |
| Export | ONNX opset 11, simplify | nncase hỗ trợ tốt |
| nncase | 2.10.0 + .NET 7 | CanMV K230 docs khuyến nghị |
| Input type | uint8 | Camera K230 cho uint8 |
| Output type | float32 | Box + score cần chính xác |
| Conf thresh | 0.3 | Cân bằng P/R cho drone |
| NMS thresh | 0.7 | Tiêu chuẩn YOLO |

---

## 7. Mở rộng trong tương lai

### Cải thiện mô hình
- **Thêm data class hiếm** (`spring_weevil`, `citrus_longhorned_beetle`) → AP class đó tăng.
- **Train lại với imgsz 800** (nếu chấp nhận FPS thấp hơn trên K230) → vật thể nhỏ rõ hơn.
- **Thử YOLOv8s** nếu K230 có version RAM cao hơn → chính xác hơn, chậm hơn.

### Drone integration
- **GPS logging**: box + vị trí GPS → bản đồ nhiệt vùng bệnh.
- **Auto-land**: phát hiện bệnh nghiêm trọng → drone hạ cánh cảnh báo.
- **Loa/buzzer**: phát âm thanh khi thấy pest cụ thể.

### Edge cases
- **Ban đêm**: camera IR + model train trên ảnh IR.
- **Mưa/sương**: augment thêm noise + độ ẩm.
- **Lá chồng lá**: thử mô hình segmentation (YOLOv8-seg) thay vì detection.

---

## 8. Tài liệu tham khảo

- **Ultralytics docs**: `docs.ultralytics.com/` — YOLOv8 đầy đủ
- **CanMV K230**: `developer.canaan-creative.com/k230/` — tutorial CanMV
- **nncase**: `github.com/kendryte/nncase` — releases + docs
- **Roboflow**: `roboflow.com` — annotate + export dataset
- **Netron**: `netron.app` — visualize ONNX/kmodel

---

## 9. Index tài liệu docs

| Doc | Nội dung |
|---|---|
| [01_yolo_architecture.md](01_yolo_architecture.md) | Kiến trúc YOLOv8: backbone/neck/head, anchor-free, input/output shape |
| [02_training_process.md](02_training_process.md) | Training loop, loss, optimizer, augmentation, metrics, backup |
| [03_data_prep.md](03_data_prep.md) | Dataset lifecycle, YOLO label format, split, class imbalance |
| [04_export_deploy.md](04_export_deploy.md) | Export ONNX, nncase compile kmodel, deploy K230, troubleshooting |
| [05_project_walkthrough.md](05_project_walkthrough.md) | Bản đồ tổng thể (file này) |

Đọc theo thứ tự 01 → 05 để hiểu toàn bộ project từ kiến trúc đến deploy.
