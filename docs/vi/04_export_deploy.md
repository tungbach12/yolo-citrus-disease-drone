# 04 — Export & Deploy lên K230

> Tài liệu về chuỗi export: `best.pt` → `best.onnx` → `best.kmodel`, và deploy lên CanMV K230. Tham chiếu [scripts/convert_to_kmodel.py](../../scripts/convert_to_kmodel.py) và [scripts/k230_yolov8_det.py](../../scripts/k230_yolov8_det.py).

## 1. Vì sao phải export?

`best.pt` (PyTorch checkpoint) **không chạy được trên K230** vì:
- K230 có KPU ( Kendryte Process Unit) — phần cứng专属 cho AI, không hiểu PyTorch.
- PyTorch runtime nặng (~1GB), K230 chỉ có ~512MB RAM.
- K230 chạy CanMV (MicroPython), không có Python CPython đầy đủ.

Chuỗi chuyển đổi:
```
best.pt (PyTorch)
   ↓ export
best.onnx (ONNX — trung gian phổ quát)
   ↓ compile nncase
best.kmodel (KPU binary — chạy được trên K230)
```

---

## 2. ONNX — Open Neural Network Exchange

ONNX là **format trung gian** — mô hình độc lập framework (PyTorch, TensorFlow, MXNet đều export được ONNX).

**Vì sao dùng ONNX làm trung gian, không compile thẳng .pt → kmodel?**
- nncase hỗ trợ ONNX tốt nhất (đồ thị tĩnh, rõ ràng).
- ONNX có công cụ inspect (netron.app), debug dễ.
- Nếu sau này đổi framework train (TF, JAX) → vẫn export ONNX → cùng pipeline deploy.

### Export từ ultralytics

Cell-13 notebook:
```python
best = YOLO(f"{RESULTS_DIR}/best.pt")
best.export(format="onnx", imgsz=IMGSZ, opset=11, simplify=True)
```

Project chạy local trên laptop (CPU):
```python
m = YOLO("models/best.pt")
m.export(format="onnx", imgsz=640, opset=11, simplify=True)
```

### Các tham số

| Tham số | Giá trị | Vì sao |
|---|---|---|
| `format` | `"onnx"` | Trung gian cho nncase |
| `imgsz` | `640` | **Phải khớp** imgsz train + `model_input_size` K230 |
| `opset` | `11` | nncase 2.10 hỗ trợ tốt opset 11. Opset quá mới (15+) có operator nncase chưa support |
| `simplify` | `True` | Dùng onnxslim tối ưu đồ thị — gộp layer, bỏ layer dư → kmodel nhỏ hơn, nhanh hơn |

### Kết quả

```
ONNX saved as 'models/best.onnx' (11.7 MB)
Model summary (fused): 73 layers, 3,013,253 parameters, 8.1 GFLOPs
```

> **Kiểm tra đồ thị:** Mở `models/best.onnx` trên `netron.app` → xem từng layer, input/output shape. Rất hữu ích khi debug "operator không support".

---

## 3. nncase — compile ONNX → kmodel

### nncase là gì

**nncase** là compiler của Kendryte (Canaan) — biên dịch mô hình neural sang binary chạy trên KPU của K230 (và K210). Pipeline:

```
ONNX → [nncase import] → IR → [optimize + quantize] → KPU binary → .kmodel
```

### Cài đặt (Windows)

```powershell
pip install nncase==2.10.0
# nncase_kpu riêng (KPU ops không có trong nncase core)
curl -L -O https://github.com/kendryte/nncase/releases/download/v2.10.0/nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
pip install nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
```

**Phụ thuộc .NET 7:** nncase 2.10 dùng .NET runtime → cần **.NET 7.0 x64 Desktop Runtime**. Máy có .NET 9 không đủ — phải cài thêm .NET 7. Nếu thiếu → lỗi `Failed to initialize hostfxr: 0x80008096`.

### API 2.10.0 — khác bản cũ

nncase 2.10.0 API đổi đáng kể so với tutorial cũ. Script [scripts/convert_to_kmodel.py](../../scripts/convert_to_kmodel.py) đã viết đúng API mới:

```python
import nncase

# (1) Batch size về 1 — K230 infer từng frame
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

# (2) Compile options — truyền VÀO constructor Compiler
compile_options = nncase.CompileOptions()
compile_options.target = "k230"
compile_options.input_type = "uint8"
compile_options.input_layout = "NCHW"
compile_options.output_type = "float32"
compile_options.output_layout = "NCHW"
compile_options.preprocess = False
compile_options.input_shape = [1, 3, 640, 640]

compiler = nncase.Compiler(compile_options)  # ← options ở đây, KHÔNG phải compile()

# (3) Import ONNX — nhận BYTES + ImportOptions (không phải path + layout)
with open("best_bs1.onnx", "rb") as f:
    model_bytes = f.read()
import_options = nncase.ImportOptions()
compiler.import_onnx(model_bytes, import_options)

# (4) Compile + ghi kmodel
compiler.compile()
kmodel = compiler.gencode_tobytes()
with open("best.kmodel", "wb") as f:
    f.write(kmodel)
```

### Giải thích options

| Option | Giá trị | Vì sao |
|---|---|---|
| `target` | `"k230"` | Biên dịch cho KPU K230 (không phải K210) |
| `input_type` | `"uint8"` | Camera K230 cho ảnh uint8 (0-255). Để uint8 → nncase tự dequantize trong KPU |
| `output_type` | `"float32"` | Output cần chính xác (box + score) → giữ float32, không quantize output |
| `input_layout` / `output_layout` | `"NCHW"` | Batch/Channel/Height/Width — chuẩn ONNX |
| `preprocess` | `False` | Tự xử lý preprocess (resize, letterbox) trong Python K230, không nhờ nncase |
| `input_shape` | `[1,3,640,640]` | Batch 1, 3 kênh RGB, 640×640 |

> **Vì sao preprocess=False?** nncase có preprocess built-in (resize, mean/std, letterbox) nhưng khó kiểm soát + phiên bản khác nhau hành vi khác. Project tự làm preprocess trong `k230_yolov8_det.py` bằng PipeLine CanMV → rõ ràng, debug dễ.

### Kết quả

```
best.kmodel (11883.2 KB ≈ 11.6 MB)
```

---

## 4. Deploy lên K230

### Copy file

2 file cần copy lên thẻ SD K230 (thư mục `/sdcard`):
1. `models/best.kmodel` → `/sdcard/best.kmodel`
2. `scripts/k230_yolov8_det.py` → `/sdcard/k230_yolov8_det.py`

CanMV IDE mở `k230_yolov8_det.py` → Connect → Run.

### Script K230 — luồng chạy

[scripts/k230_yolov8_det.py](../../scripts/k230_yolov8_det.py):

```python
from libs.PipeLine import PipeLine, ScopedTiming
from libs.YOLO import YOLOv8

kmodel_path = "/sdcard/best.kmodel"
labels = [39 class citrus...]
model_input_size = [640, 640]

rgb888p_size = [1280, 720]    # camera capture
display_size = [800, 480]     # màn LCD

pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode="lcd")
pl.create()

yolo = YOLOv8(
    task_type="detect", mode="video",
    kmodel_path=kmodel_path, labels=labels,
    rgb888p_size=rgb888p_size, model_input_size=model_input_size,
    display_size=display_size,
    conf_thresh=0.3, nms_thresh=0.7,
    debug_mode=0,
)
yolo.config_preprocess()

while True:
    img = pl.get_frame()           # camera → frame RGB888 1280×720
    res = yolo.run(img)            # preprocess + infer + postprocess
    yolo.draw_result(res, pl.osd_img)  # vẽ box + label lên OSD
    pl.show_image()                # hiển thị màn
```

### PipeLine — CanMV

`libs.PipeLine` (có sẵn trong CanMV firmware) quản lý luồng camera → AI → display:
- Capture frame từ camera (sensor) ở `rgb888p_size`.
- Cung cấp OSD (On-Screen Display) layer để vẽ kết quả.
- Hiển thị lên màn LCD hoặc HDMI.

### YOLOv8 class — CanMV

`libs.YOLOv8` (CanMV built-in) wrap toàn bộ:
- **config_preprocess**: set up resize 1280×720 → 640×640 + letterbox (padding giữ tỷ lệ).
- **run(img)**: 
  1. Preprocess ảnh đầu vào → tensor uint8 `(1,3,640,640)`.
  2. Nạp kmodel, infer → output `(1,43,8400)` float32.
  3. Decode box + lọc conf + NMS → list box cuối.
- **draw_result**: vẽ box + class label + confidence lên OSD.

### Tham số quan trọng

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `kmodel_path` | `/sdcard/best.kmodel` | Đường dẫn kmodel trên thẻ SD |
| `labels` | 39 class citrus | **Phải khớp** `names` data.yaml + output model |
| `model_input_size` | `[640, 640]` | **Phải khớp** imgsz export ONNX |
| `rgb888p_size` | `[1280, 720]` | Camera capture resolution |
| `display_size` | `[800, 480]` | Màn LCD K230 |
| `conf_thresh` | `0.3` | Bỏ box confidence < 0.3 |
| `nms_thresh` | `0.7` | NMS IoU threshold |
| `debug_mode` | `0` | 0=silent, 1=print timing, 2=full debug |

> **Nếu labels sai thứ tự:** Mô hình output class_id 7 (citrus_canker) nhưng labels[7] = "black_aphid" → hiển thị sai hoàn toàn. **Kiểm tra kỹ** thứ tự `labels` trong script khớp `names` trong data.yaml. Project đã đồng bộ.

---

## 5. Troubleshooting deploy

### Lỗi "kmodel not found"
- Kiểm tra `kmodel_path` đúng, file có trên thẻ SD.
- Thẻ SD mount ở `/sdcard` — một số board K230 có thể `/sdcard1` hoặc `/`. Dùng `os.listdir('/')` trong CanMV để kiểm tra.

### Lỗi "shape mismatch"
- `model_input_size` ≠ imgsz export ONNX → sửa 1 trong 2 cho khớp (khuyến nghị sửa script, không re-export).
- Kmodel batch > 1 nhưng K230 infer 1 frame → re-compile với `input_shape[0]=1`.

### FPS thấp (< 5)
- Giảm `rgb888p_size` (vd 640×480) → ít data hơn, nhanh hơn.
- Tăng `conf_thresh` (vd 0.5) → ít box hơn, postprocess nhanh.
- Kiểm tra `debug_mode=0` (mode debug chậm).

### Box lệch / sai vị trí
- Letterbox sai — `config_preprocess` phải dùng cùng tỷ lệ scale cho cả X và Y.
- `rgb888p_size` ≠ display ratio → box vẽ sai tỷ lệ trên OSD. Đảm bảo PipeLine xử lý đúng.

### Confidence toàn thấp
- Mô hình chưa đủ train → re-train thêm epoch hoặc thêm data.
- Ảnh K230 khác điều kiện dataset (sáng, góc) → thu thêm data đa dạng.

---

## 6. Lưu ý env Windows (nncase)

- **.NET 7 bắt buộc** — không có → nncase import fail. Cài `.NET 7.0 x64 Desktop Runtime`.
- Cảnh báo `NNCASE_PLUGIN_PATH is not set` → **bình thường**, bỏ qua.
- nncase 2.10 không tương thích ngược API cũ — script dùng `Compiler(options)` + `import_onnx(bytes, ImportOptions)`, không phải `Compiler()` + `import_onnx(path, layout)`.

---

## 7. Tổng kết

| Bước | Tool | Output | Kích thước |
|---|---|---|---|
| Train | Ultralytics + Kaggle GPU | `best.pt` | 5.4 MB |
| Export ONNX | `ultralytics export` (CPU OK) | `best.onnx` | 11.7 MB |
| Compile kmodel | nncase 2.10.0 + .NET 7 | `best.kmodel` | 11.6 MB |
| Deploy | copy 2 file → SD K230 | chạy CanMV | — |

**File liên quan:**
- [scripts/convert_to_kmodel.py](../../scripts/convert_to_kmodel.py) — compile ONNX → kmodel (nncase 2.10.0 API)
- [scripts/k230_yolov8_det.py](../../scripts/k230_yolov8_det.py) — script chạy trên K230 (39 labels)
- [models/best.onnx](../../models/best.onnx) — bản trung gian
- [models/best.kmodel](../../models/best.kmodel) — bản deploy cuối

**Tiếp theo:** [05_project_walkthrough.md](05_project_walkthrough.md) — bản đồ tổng thể project + cách chạy từng phần.
