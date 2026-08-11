# 01 — YOLO & YOLOv8 Architecture

> Self-study guide to the YOLO family of object detection models, focused on YOLOv8 (the `n` variant this project uses to deploy on the K230 drone). Read this before [02_training_process.md](02_training_process.md).

## 1. What is YOLO — "You Only Look Once"

**YOLO** (You Only Look Once) is a family of **one-stage** object detection models. The core idea: pass the image through the network **once** and get back the full list of objects — locations and classes — in a single pass.

### One-stage vs Two-stage

| | Two-stage (R-CNN, Faster R-CNN) | One-stage (YOLO, SSD) |
|---|---|---|
| How it works | Step 1: find ~2000 candidate regions (region proposals). Step 2: classify each. | Pass the whole image through the network → all boxes + classes at once. |
| Accuracy | Slightly higher (especially for small objects) | Slightly lower |
| Speed | Slow (tens of FPS) | Fast (hundreds of FPS) |
| Best for | Medical imaging, satellite — accuracy matters more than time | Drones, cameras, self-driving — realtime needed |

**Why YOLO for a drone?** A drone streams ~30 frames per second. If the model is slower than the frame rate, latency accumulates → the drone reacts late → loses the target. YOLO is fast enough to run **realtime** on weak hardware (K230, Raspberry Pi).

> **The trade-off:** YOLOv8n on the K230 runs at ~10-20 FPS — enough for a slow-flying drone scanning a citrus orchard. If you need absolute precision (very early disease, a 5px lesion) a two-stage model is more accurate, but it cannot run realtime on the K230.

---

## 2. YOLOv8 Architecture — 3 parts: Backbone → Neck → Head

Every modern detection model has three blocks:

```
Input image (640×640)
      ↓
  [Backbone]  — extract features from the image
      ↓
  [Neck]      — fuse features across multiple scales
      ↓
  [Head]      — produce predictions: box + class for each object
```

### 2.1 Backbone — CSPDarknet

The backbone is the "observer": it takes the raw image and extracts **features** at multiple levels of abstraction:
- **Early layers**: detect edges, corners, textures (fine detail).
- **Late layers**: detect high-level semantic structure (complex objects: an insect, a lesion pattern).

YOLOv8 uses **CSPDarknet** (Cross-Stage Partial Darknet). "CSP" = split the feature map into two branches, process them in parallel, then merge → **reduces redundant computation** while preserving information. This is why YOLOv8n is light but still powerful.

### 2.2 Neck — PANet (FPN + PAN)

The neck handles **multi-scale detection** — finding both large and small objects in the same image:
- **FPN** (Feature Pyramid Network): passes information from late layers (big picture, context) down to early layers (detail) → helps detect large objects.
- **PAN** (Path Aggregation Network): passes information back up from early to late → preserves fine detail for small objects.

On a drone looking down from above, objects (diseased citrus leaves) are usually **very small** relative to a 640×640 image → PANet is critical. This is why YOLOv8 detects small objects much better than older YOLO versions.

### 2.3 Head — Decoupled, Anchor-free

The **head** is the "decision maker": it takes the fused features and produces the final predictions.

**Decoupled head** (split): YOLOv8 separates the **box** prediction (location) and **class** prediction into two independent branches. Older versions (YOLOv5) combined them → gradient conflict, harder to converge. Splitting them lets each branch learn its own task well.

**Anchor-free**: this is YOLOv8's biggest change. See below.

---

## 3. What "anchor-free" means — and why YOLOv8 dropped anchors

### What anchor boxes were (the old way)

YOLOv5 and earlier used **anchor boxes** — preset "template frames" of various sizes (e.g. `[10,13], [16,30], [33,23]` for small objects). The model learned to **adjust** an anchor to fit an object, rather than predicting a box from scratch.

**Problems:**
- Anchors had to be **chosen upfront** by clustering ground-truth sizes → switching datasets means recomputing anchors.
- On small or awkward datasets, wrong anchors → weak model.
- Extra processing complexity.

### YOLOv8: anchor-free

YOLOv8 predicts **directly** 4 values: the distance from the center of a grid cell to each of the 4 edges (top/bottom/left/right) of the box. No anchors needed.

**Benefits:**
- No anchor tuning — switch datasets without changing anything.
- Detects objects of any size (not limited by anchor templates).
- Simpler and faster.

**This is why you do not see `anchors` in this project's `data.yaml` — YOLOv8 does not use them.**

---

## 4. Model sizes — n / s / m / l / x

YOLOv8 comes in 5 sizes, all sharing the same architecture but with different depth (number of layers) and width (number of channels):

| Variant | Params | GFLOPs | Purpose |
|---|---|---|---|
| **YOLOv8n** (nano) | 3.0M | 8.1 | Edge devices (K230, Pi, mobile) |
| YOLOv8s (small) | 11.2M | 28.6 | Light server, Jetson |
| YOLOv8m (medium) | 25.9M | 78.9 | Workstation GPU |
| YOLOv8l (large) | 43.7M | 165.2 | Powerful GPU |
| YOLOv8x (xlarge) | 68.2M | 257.8 | Server, competition |

### Why this project uses `n`

The K230 has a limited KPU memory budget (~1GB region) and limited compute. A larger model → a larger kmodel → may not load, or runs too slowly.

This project: `best.pt` = 5.4 MB, `best.kmodel` = 11.6 MB. On the K230 it runs at ~10-20 FPS. If we used `s` → kmodel ~50MB, might not load or only 2-3 FPS.

> **Trade-off:** `n` has fewer parameters → less accurate than `s/m`. But on the K230 there is no real choice — `n` is the sweet spot between "runs" and "runs fast".

**Verified in practice:** when loading `models/best.pt` with ultralytics, the summary output is:
```
Model summary (fused): 73 layers, 3,013,253 parameters, 0 gradients, 8.1 GFLOPs
```
→ matches the YOLOv8n spec exactly.

---

## 5. Input shape & Output shape — reading the numbers

### Input: `(1, 3, 640, 640)` — BCHW

| Dim | Meaning |
|---|---|
| `1` | Batch size — 1 image per inference (the K230 runs frame by frame) |
| `3` | 3 color channels RGB |
| `640` | Height (pixels) |
| `640` | Width (pixels) |

**BCHW** = Batch/Channel/Height/Width — the axis order PyTorch/ONNX uses. (OpenCV gives images in HWC, so you must transpose before feeding them in.)

`640` is `IMGSZ` in the notebook — it **must match** across training, ONNX export, and `model_input_size` in `k230_yolov8_det.py`. Change one without the others → the model breaks.

### Output: `(1, 43, 8400)`

This is the interesting part. Reading right-to-left the way the model "thinks":

**`8400`** = total grid cells across 3 scales. YOLOv8 divides the image into grids at 3 resolutions:
- 80×80 (stride 8) — detects **small** objects (a diseased leaf from afar)
- 40×40 (stride 16) — **medium** objects
- 20×20 (stride 32) — **large** objects

```
80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 = 8400 cells
```
Each cell predicts **1 object** (anchor-free) → up to 8400 boxes per image.

**`43`** = values each cell predicts:
- `4`: box coordinates — distances from the cell center to the 4 edges (top/bottom/left/right)
- `39`: scores for the 39 citrus classes (probability of each disease/pest)

```
4 + 39 = 43
```

**`1`** = batch size.

> **Consequence:** If you switch to a dataset with N classes, the output shape automatically becomes `(1, 4+N, 8400)`. No code change needed. This is why YOLOv8 is "flexible" — only `data.yaml`'s `nc` has to be right.

---

## 6. Post-processing — from raw output to final boxes

The output `(1, 43, 8400)` is raw — three steps turn it into usable boxes:

1. **Decode boxes**: the raw 4 values (cell-center-to-edge distances) → `xyxy` pixel coordinates (x1,y1,x2,y2).
2. **Confidence filter**: drop any cell whose max(class_score) is below `conf_thresh` (e.g. 0.3). A few dozen boxes remain.
3. **NMS** (Non-Maximum Suppression): multiple cells may detect **the same object** → overlapping boxes. NMS keeps the highest-confidence box and discards the rest (those with IoU > `nms_thresh`, e.g. 0.7).

In `k230_yolov8_det.py`:
```python
confidence_threshold = 0.3
nms_threshold = 0.7
```

**Why NMS is needed:** With anchor-free + 3 scales, a single red wax scale insect might be detected by an 80×80 cell, a 40×40 cell, and a 20×20 cell → 3 overlapping boxes. NMS merges them into 1.

---

## 7. Summary — the data flow diagram

```
K230 camera (1280×720)
   ↓ resize + letterbox
640×640 RGB image
   ↓ [Backbone CSPDarknet]
Multi-scale feature maps
   ↓ [Neck PANet]
Fused features (80/40/20 cells)
   ↓ [Head anchor-free]
Raw output (1, 43, 8400)
   ↓ decode + conf filter + NMS
Final boxes: [x1,y1,x2,y2, class_id, confidence] × N
   ↓ draw on OSD
Display on 800×480 LCD
```

**Related files in this project:**
- `models/best.pt` — PyTorch checkpoint (runs on laptop via ultralytics)
- `models/best.onnx` — intermediate, same architecture, runs via onnxruntime
- `models/best.kmodel` — compiled for the K230 KPU, same architecture but quantized and hardware-optimized

---

## 8. References

- YOLOv8 docs: `docs.ultralytics.com/models/yolov8/`
- Anchor-free explanation: `docs.ultralytics.com/anchor-free-detection/`
- Netron (visualize the ONNX graph): `netron.app` — open `models/best.onnx` to inspect the graph layer by layer

**Next:** [02_training_process.md](02_training_process.md) — the training process, loss functions, optimizer, augmentation, metrics.
