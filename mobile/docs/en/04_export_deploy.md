# 04 — Export & Deploy to the K230

> Guide to the export chain: `best.pt` → `best.onnx` → `best.kmodel`, and deployment on the CanMV K230. References [scripts/convert_to_kmodel.py](../../scripts/convert_to_kmodel.py) and [scripts/k230_yolov8_det.py](../../scripts/k230_yolov8_det.py).

## 1. Why export?

`best.pt` (a PyTorch checkpoint) **cannot run on the K230** because:
- The K230 has a KPU (Kendryte Process Unit) — AI-specific hardware that does not understand PyTorch.
- The PyTorch runtime is heavy (~1GB); the K230 only has ~512MB RAM.
- The K230 runs CanMV (MicroPython), not full CPython.

The conversion chain:
```
best.pt (PyTorch)
   ↓ export
best.onnx (ONNX — universal interchange)
   ↓ compile nncase
best.kmodel (KPU binary — runs on the K230)
```

---

## 2. ONNX — Open Neural Network Exchange

ONNX is an **intermediate format** — a model representation independent of any framework (PyTorch, TensorFlow, MXNet can all export to ONNX).

**Why use ONNX as the intermediate, instead of compiling .pt → kmodel directly?**
- nncase supports ONNX best (static graph, clear semantics).
- ONNX has inspection tools (netron.app), easy to debug.
- If you later switch training frameworks (TF, JAX) → still export ONNX → same deployment pipeline.

### Exporting from ultralytics

Notebook cell-13:
```python
best = YOLO(f"{RESULTS_DIR}/best.pt")
best.export(format="onnx", imgsz=IMGSZ, opset=11, simplify=True)
```

This project ran it locally on the laptop (CPU):
```python
m = YOLO("models/best.pt")
m.export(format="onnx", imgsz=640, opset=11, simplify=True)
```

### Parameters

| Param | Value | Why |
|---|---|---|
| `format` | `"onnx"` | Intermediate for nncase |
| `imgsz` | `640` | **Must match** training imgsz + K230 `model_input_size` |
| `opset` | `11` | nncase 2.10 supports opset 11 well. Newer opsets (15+) may use operators nncase does not support yet |
| `simplify` | `True` | Uses onnxslim to optimize the graph — fold layers, remove redundancy → smaller, faster kmodel |

### Result

```
ONNX saved as 'models/best.onnx' (11.7 MB)
Model summary (fused): 73 layers, 3,013,253 parameters, 8.1 GFLOPs
```

> **Inspect the graph:** Open `models/best.onnx` on `netron.app` → see every layer, input/output shapes. Very useful when debugging "operator not supported".

---

## 3. nncase — compiling ONNX → kmodel

### What is nncase

**nncase** is Kendryte's (Canaan's) compiler — it compiles neural network models into binaries that run on the K230's KPU (and the K210). Pipeline:

```
ONNX → [nncase import] → IR → [optimize + quantize] → KPU binary → .kmodel
```

### Installation (Windows)

```powershell
pip install nncase==2.10.0
# nncase_kpu is separate (KPU ops are not in the nncase core)
curl -L -O https://github.com/kendryte/nncase/releases/download/v2.10.0/nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
pip install nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
```

**.NET 7 dependency:** nncase 2.10 uses the .NET runtime → requires **.NET 7.0 x64 Desktop Runtime**. Having only .NET 9 is not enough — you must install .NET 7 alongside it. If missing → error `Failed to initialize hostfxr: 0x80008096`.

### The 2.10.0 API — different from old tutorials

nncase 2.10.0's API changed significantly from older tutorials. The script [scripts/convert_to_kmodel.py](../../scripts/convert_to_kmodel.py) is written for the new API:

```python
import nncase

# (1) Set batch size to 1 — the K230 infers frame by frame
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

# (2) Compile options — passed INTO the Compiler constructor
compile_options = nncase.CompileOptions()
compile_options.target = "k230"
compile_options.input_type = "uint8"
compile_options.input_layout = "NCHW"
compile_options.output_type = "float32"
compile_options.output_layout = "NCHW"
compile_options.preprocess = False
compile_options.input_shape = [1, 3, 640, 640]

compiler = nncase.Compiler(compile_options)  # ← options here, NOT in compile()

# (3) Import ONNX — takes BYTES + ImportOptions (not path + layout)
with open("best_bs1.onnx", "rb") as f:
    model_bytes = f.read()
import_options = nncase.ImportOptions()
compiler.import_onnx(model_bytes, import_options)

# (4) Compile + write kmodel
compiler.compile()
kmodel = compiler.gencode_tobytes()
with open("best.kmodel", "wb") as f:
    f.write(kmodel)
```

### Options explained

| Option | Value | Why |
|---|---|---|
| `target` | `"k230"` | Compile for the K230 KPU (not K210) |
| `input_type` | `"uint8"` | The K230 camera produces uint8 images (0-255). Using uint8 → nncase handles dequantization in the KPU |
| `output_type` | `"float32"` | Outputs need precision (boxes + scores) → keep float32, do not quantize outputs |
| `input_layout` / `output_layout` | `"NCHW"` | Batch/Channel/Height/Width — ONNX standard |
| `preprocess` | `False` | We handle preprocessing (resize, letterbox) ourselves in the K230 Python script, not in nncase |
| `input_shape` | `[1,3,640,640]` | Batch 1, 3 RGB channels, 640×640 |

> **Why preprocess=False?** nncase has built-in preprocessing (resize, mean/std, letterbox) but it is hard to control and behaves differently across versions. This project does preprocessing manually in `k230_yolov8_det.py` via the CanMV PipeLine → clear, easy to debug.

### Result

```
best.kmodel (11883.2 KB ≈ 11.6 MB)
```

---

## 4. Deploying to the K230

### Copy files

Two files need to be copied to the K230's SD card (in `/sdcard`):
1. `models/best.kmodel` → `/sdcard/best.kmodel`
2. `scripts/k230_yolov8_det.py` → `/sdcard/k230_yolov8_det.py`

Open `k230_yolov8_det.py` in the CanMV IDE → Connect → Run.

### The K230 script — execution flow

[scripts/k230_yolov8_det.py](../../scripts/k230_yolov8_det.py):

```python
from libs.PipeLine import PipeLine, ScopedTiming
from libs.YOLO import YOLOv8

kmodel_path = "/sdcard/best.kmodel"
labels = [39 citrus classes...]
model_input_size = [640, 640]

rgb888p_size = [1280, 720]    # camera capture
display_size = [800, 480]     # LCD screen

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
    img = pl.get_frame()           # camera → RGB888 frame 1280×720
    res = yolo.run(img)            # preprocess + infer + postprocess
    yolo.draw_result(res, pl.osd_img)  # draw boxes + labels on OSD
    pl.show_image()                # display on screen
```

### PipeLine — CanMV

`libs.PipeLine` (bundled with CanMV firmware) manages the camera → AI → display pipeline:
- Captures frames from the camera sensor at `rgb888p_size`.
- Provides an OSD (On-Screen Display) layer to draw results on.
- Displays to the LCD or HDMI.

### The YOLOv8 class — CanMV

`libs.YOLOv8` (CanMV built-in) wraps everything:
- **config_preprocess**: sets up resize 1280×720 → 640×640 + letterbox (padding to preserve aspect ratio).
- **run(img)**:
  1. Preprocess the input image → uint8 tensor `(1,3,640,640)`.
  2. Load the kmodel, infer → output `(1,43,8400)` float32.
  3. Decode boxes + confidence filter + NMS → final box list.
- **draw_result**: draws boxes + class labels + confidence scores on the OSD.

### Key parameters

| Param | Value | Meaning |
|---|---|---|
| `kmodel_path` | `/sdcard/best.kmodel` | Path to kmodel on the SD card |
| `labels` | 39 citrus classes | **Must match** `names` in data.yaml + model output |
| `model_input_size` | `[640, 640]` | **Must match** the imgsz used in ONNX export |
| `rgb888p_size` | `[1280, 720]` | Camera capture resolution |
| `display_size` | `[800, 480]` | K230 LCD screen |
| `conf_thresh` | `0.3` | Drop boxes with confidence < 0.3 |
| `nms_thresh` | `0.7` | NMS IoU threshold |
| `debug_mode` | `0` | 0=silent, 1=print timing, 2=full debug |

> **If labels are in the wrong order:** The model outputs class_id 7 (citrus_canker) but labels[7] = "black_aphid" → the display is completely wrong. **Double-check** that the `labels` order in the script matches `names` in data.yaml. This project has them synced.

---

## 5. Deployment troubleshooting

### "kmodel not found" error
- Check that `kmodel_path` is correct and the file exists on the SD card.
- The SD card mounts at `/sdcard` — some K230 boards may use `/sdcard1` or `/`. Use `os.listdir('/')` in CanMV to check.

### "shape mismatch" error
- `model_input_size` ≠ ONNX export imgsz → fix one to match the other (prefer fixing the script, not re-exporting).
- Kmodel batch > 1 but the K230 infers 1 frame → recompile with `input_shape[0]=1`.

### Low FPS (< 5)
- Reduce `rgb888p_size` (e.g. 640×480) → less data, faster.
- Raise `conf_thresh` (e.g. 0.5) → fewer boxes, faster postprocessing.
- Make sure `debug_mode=0` (debug modes are slow).

### Boxes are offset / wrong position
- Letterbox is wrong — `config_preprocess` must use the same scale for X and Y.
- `rgb888p_size` ≠ display aspect ratio → boxes draw with wrong proportions on OSD. Ensure the PipeLine handles this correctly.

### All confidences are low
- The model is undertrained → retrain with more epochs or more data.
- The K230 image conditions differ from the dataset (lighting, angle) → collect more diverse data.

---

## 6. Notes on the Windows environment (nncase)

- **.NET 7 is required** — without it, `import nncase` fails. Install `.NET 7.0 x64 Desktop Runtime`.
- The `NNCASE_PLUGIN_PATH is not set` warning → **benign**, ignore it.
- nncase 2.10 is not backward-compatible with the old API — the script uses `Compiler(options)` + `import_onnx(bytes, ImportOptions)`, not `Compiler()` + `import_onnx(path, layout)`.

---

## 7. Summary

| Step | Tool | Output | Size |
|---|---|---|---|
| Train | Ultralytics + Kaggle GPU | `best.pt` | 5.4 MB |
| Export ONNX | `ultralytics export` (CPU OK) | `best.onnx` | 11.7 MB |
| Compile kmodel | nncase 2.10.0 + .NET 7 | `best.kmodel` | 11.6 MB |
| Deploy | copy 2 files → K230 SD card | run in CanMV | — |

**Related files:**
- [scripts/convert_to_kmodel.py](../../scripts/convert_to_kmodel.py) — compile ONNX → kmodel (nncase 2.10.0 API)
- [scripts/k230_yolov8_det.py](../../scripts/k230_yolov8_det.py) — the script that runs on the K230 (39 labels)
- [models/best.onnx](../../models/best.onnx) — intermediate
- [models/best.kmodel](../../models/best.kmodel) — final deploy artifact

**Next:** [05_project_walkthrough.md](05_project_walkthrough.md) — the overall project map + how to run each part.
