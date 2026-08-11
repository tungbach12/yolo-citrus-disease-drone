# 05 — Project Walkthrough (The full map)

> Summary document: the project goal, the end-to-end pipeline, the directory structure, how to run each part, and lessons learned along the way. Read this after 01-04.

## 1. Project goal

**An agricultural drone that detects citrus diseases in realtime.** The drone (K230 board) flies over a citrus orchard → the camera detects leaves/diseases/pests → displays a box + disease name on screen → the farmer knows which area needs treatment.

**Technical requirements:**
- Detect **39 classes** (diseases + pests + beneficial insects + healthy leaves).
- **Realtime** on the K230 (~10-20 FPS).
- Test on a **laptop webcam** before deploying to the drone (using `best.pt`).

---

## 2. End-to-end pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATASET (Roboflow)                                        │
│    Citrus Disease Detection v1 — 39 classes, YOLOv8 format   │
│    downloaded directly in the notebook                       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAIN (Kaggle GPU T4)                                     │
│    notebooks/train_yolov8n_drone_kaggle.ipynb               │
│    YOLOv8n, 50 epochs, patience 15, auto-backup each epoch   │
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
│   realtime   │ → 800×480 LCD
└──────────────┘
```

---

## 3. Directory structure

```
E:\Dev\Projects\Yolo\
├── models/
│   ├── best.pt              # trained weights (laptop webcam + ONNX export source)
│   ├── best.onnx            # intermediate ONNX (kmodel compile source)
│   ├── best.kmodel          # KPU binary for K230 deployment
│   └── yolov8n.pt           # COCO pretrained (transfer learning source)
├── notebooks/
│   ├── train_yolov8n_drone_kaggle.ipynb   # train on Kaggle (fixed)
│   └── train_yolov8n_drone.ipynb          # train on Colab (variant)
├── scripts/
│   ├── convert_to_kmodel.py   # ONNX → kmodel (nncase 2.10.0)
│   ├── realtime_cam.py        # laptop webcam test
│   ├── k230_yolov8_det.py     # runs on the K230
│   └── rebuild_pt.py          # rebuild .pt from Kaggle web-UI zip (reference)
├── data/
│   └── citrus/                # dataset metadata only (no images)
│       ├── data.yaml          # 39 class names + config
│       ├── README.dataset.txt
│       └── README.roboflow.txt
└── docs/
    ├── vi/                    # Vietnamese version
    │   ├── 01_yolo_architecture.md
    │   ├── 02_training_process.md
    │   ├── 03_data_prep.md
    │   ├── 04_export_deploy.md
    │   └── 05_project_walkthrough.md
    └── en/                    # English version
        ├── 01_yolo_architecture.md
        ├── 02_training_process.md
        ├── 03_data_prep.md
        ├── 04_export_deploy.md
        └── 05_project_walkthrough.md   ← you are here (the map)
```

---

## 4. How to run each part

### 4.1 Training (on Kaggle)

1. Upload `notebooks/train_yolov8n_drone_kaggle.ipynb` to Kaggle.
2. Settings: Accelerator = **GPU T4 x2**, Internet = **ON**.
3. Run cells 0 → 14 in order.
4. When done, download `best.zip` (or `best_checkpoint.pt`) from the **Output** tab.

> See [02_training_process.md](02_training_process.md) to understand the training loop, losses, and metrics.

### 4.2 Export to ONNX (on the laptop)

```powershell
$env:PATH = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
python -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', imgsz=640, opset=11, simplify=True)"
```

Output: `models/best.onnx` (11.7 MB). Takes ~25s on CPU.

> See [04_export_deploy.md](04_export_deploy.md) section 2.

### 4.3 Compile the kmodel (on the laptop)

```powershell
python scripts/convert_to_kmodel.py
```

Requires: `nncase==2.10.0`, `nncase_kpu==2.10.0`, `.NET 7.0 x64` installed.

Output: `models/best.kmodel` (11.6 MB).

> See [04_export_deploy.md](04_export_deploy.md) section 3.

### 4.4 Test on the laptop webcam

```powershell
python scripts/realtime_cam.py
```

The webcam opens, draws boxes + labels in realtime. Press `q` to quit.

> Requires: `pip install ultralytics opencv-python`. On a laptop CPU → ~5-15 FPS (yolov8n is light).

### 4.5 Deploy to the K230

1. Copy `models/best.kmodel` → SD card `/sdcard/best.kmodel`.
2. Copy `scripts/k230_yolov8_det.py` → SD card `/sdcard/`.
3. Insert the SD card into the K230, open the CanMV IDE.
4. Open `k230_yolov8_det.py`, Connect (icon 1), Run (icon 2).
5. The camera + LCD run in realtime.

> See [04_export_deploy.md](04_export_deploy.md) sections 4-5.

---

## 5. Lessons learned — troubleshooting

### 5.1 Kaggle kernel ERROR — cannot download

**Situation:** The kernel `natsumeeei/capstion-v1` fell into **ERROR** state after cell-11 (preview) crashed with `FileNotFoundError`. `kernels_list_files` returned 0 files → `kaggle kernels output` (even with `--file-pattern`) hung indefinitely.

**Root cause:** The preview cell hardcoded the path `predict/predict/...jpg` — the path did not exist (ultralytics creates `predict/`, `predict2/`, etc. depending on the session) → `display(Image(filename=...))` raised FileNotFoundError → the notebook crashed before cell-13 (export ONNX) could run.

**Fix:**
- Cell-11 now wraps in try/except + uses glob to find the actual image (already fixed in the current notebook).
- **Recovery instead of re-training:** Download `best.zip` via the **Kaggle web UI** (Output tab → download icon) — the web UI still allows downloads even when the API blocks them.

**Lessons:**
1. The Kaggle API `kernels_list_files` **depends on kernel status** — ERROR state = 0 files. Do not fight the API, use the web UI.
2. Preview cells **must be fail-safe** — never let a display cell crash and block the export cells after it.
3. The **auto-backup callback** saved this project: `best_checkpoint.pt` was copied to the Output folder every epoch → even with a kernel crash, the training work up to that point was downloadable.

### 5.2 Web UI download unzips the .pt — manual rebuild

**Situation:** `best.zip` downloaded from the Kaggle web UI was **not** a pristine `.pt` file — it was the **unzipped contents** of the `.pt` (a PyTorch .pt is essentially a zip): `data.pkl`, `.data/`, `data/`, `version`, `byteorder`, ...

**Problem:** Re-zipping flat (no folder prefix) → PyTorch complains `file in archive is not in a subdirectory: .format_version`.

**Fix:** [scripts/rebuild_pt.py](../../scripts/rebuild_pt.py) — re-zips with a **folder prefix** matching the filename (`best/` for `best.pt`). PyTorch expects entries like `t/data.pkl`, not flat `data.pkl`.

**Lesson:** When a tool (Kaggle web UI) behaves "slightly off" spec, understanding the file format (.pt = a zip with structure) → you can fix it yourself instead of waiting for a tool.

### 5.3 nncase 2.10.0 API differs from old tutorials

**Situation:** The original convert script (written following an old tutorial) failed:
```
TypeError: Compiler.__init__() missing 1 required positional argument: 'compile_options'
```

**Root cause:** nncase 2.10.0 changed the API:
- `Compiler()` (no arg) → `Compiler(compile_options)`.
- `import_onnx(path, input_layout)` → `import_onnx(model_bytes, ImportOptions)`.
- `compile(options)` → `compile()` (options are in the constructor).
- `RuntimeTensorLayout("NCHW")` was removed.

**Fix:** Rewrote the script for the new API (done). Verified using `inspect.signature()` to see the real signature.

**Lesson:** Old tutorials/docs ≠ current API. When you hit a TypeError on a signature, use `python -c "import inspect; print(inspect.signature(Class.__init__))"` to see the real signature — do not trust docs.

### 5.4 nncase needs .NET 7

**Situation:** `import nncase` failed: `Failed to initialize hostfxr: 0x80008096`.

**Root cause:** nncase 2.10 was built on .NET 7. The machine had .NET 9 but **not 7** → the runtime is not backward compatible.

**Fix:** Install `.NET 7.0 x64 Desktop Runtime` (download from Microsoft).

**Lesson:** Runtime dependencies are version-specific — read the error message carefully, do not guess.

---

## 6. Finalized parameters

| Parameter | Value | Reason |
|---|---|---|
| Model | YOLOv8n | K230 constraint, needs realtime |
| Epochs | 50 | yolov8n converges fast, Kaggle does not cut off |
| Patience | 15 | Early stop if val does not improve |
| Batch | 16 | T4 VRAM 15G sweet spot |
| Imgsz | 640 | Matches train + export + K230 |
| Optimizer | SGD momentum (default) | Generalizes well for detection |
| LR schedule | Cosine (`cos_lr=True`) | Smooth, converges well |
| Augmentation | mosaic + fliplr 0.5, flipud 0 | Drone top-down, no vertical flip |
| Export | ONNX opset 11, simplify | nncase supports it well |
| nncase | 2.10.0 + .NET 7 | CanMV K230 docs recommend this |
| Input type | uint8 | K230 camera gives uint8 |
| Output type | float32 | Boxes + scores need precision |
| Conf thresh | 0.3 | Balance P/R for a drone |
| NMS thresh | 0.7 | YOLO standard |

---

## 7. Future extensions

### Improve the model
- **Add data for rare classes** (`spring_weevil`, `citrus_longhorned_beetle`) → their AP goes up.
- **Retrain with imgsz 800** (if you accept lower FPS on the K230) → small objects become clearer.
- **Try YOLOv8s** if a higher-RAM K230 variant is available → more accurate, slower.

### Drone integration
- **GPS logging**: boxes + GPS positions → a heat map of diseased areas.
- **Auto-land**: detect severe disease → the drone lands to alert.
- **Buzzer/speaker**: sound an alarm when a specific pest is seen.

### Edge cases
- **Night**: IR camera + a model trained on IR images.
- **Rain/fog**: augment with noise + humidity effects.
- **Overlapping leaves**: try a segmentation model (YOLOv8-seg) instead of detection.

---

## 8. References

- **Ultralytics docs**: `docs.ultralytics.com/` — full YOLOv8 reference
- **CanMV K230**: `developer.canaan-creative.com/k230/` — CanMV tutorials
- **nncase**: `github.com/kendryte/nncase` — releases + docs
- **Roboflow**: `roboflow.com` — annotate + export datasets
- **Netron**: `netron.app` — visualize ONNX/kmodel graphs

---

## 9. Documentation index

| Doc | Content |
|---|---|
| [01_yolo_architecture.md](01_yolo_architecture.md) | YOLOv8 architecture: backbone/neck/head, anchor-free, input/output shape |
| [02_training_process.md](02_training_process.md) | Training loop, loss, optimizer, augmentation, metrics, backup |
| [03_data_prep.md](03_data_prep.md) | Dataset lifecycle, YOLO label format, split, class imbalance |
| [04_export_deploy.md](04_export_deploy.md) | Export ONNX, nncase compile kmodel, deploy K230, troubleshooting |
| [05_project_walkthrough.md](05_project_walkthrough.md) | The overall map (this file) |

Read in order 01 → 05 to understand the entire project from architecture to deployment.
