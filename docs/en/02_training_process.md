# 02 — The YOLOv8 Training Process

> Self-study guide to the **training loop** — what the network learns, how it is optimized, when to stop. This project trained 50 epochs of YOLOv8n on a Kaggle GPU T4. References [notebooks/train_yolov8n_drone_kaggle.ipynb](../../notebooks/train_yolov8n_drone_kaggle.ipynb) cell-8.

## 1. The training loop — the core cycle

Each **epoch** (pass) trains the model on the **entire** dataset once. Each **batch** (16 images) within an epoch does:

```
for epoch in range(50):              # EPOCHS = 50
    for batch in dataset:            # 16 images per batch
        1. Forward pass      — model predicts boxes
        2. Compute loss      — compare predictions vs ground truth
        3. Backpropagation   — compute gradients (derivative of loss w.r.t. weights)
        4. Optimizer step    — update weights to reduce loss
    # end of epoch → evaluate on val → maybe save best.pt
```

Over many epochs, the loss decreases, predictions get closer to ground truth → the model "learns".

---

## 2. Forward pass

A batch of images `(16, 3, 640, 640)` flows through the network (backbone → neck → head) → output `(16, 43, 8400)` = predictions for 16 images, each with 8400 cells × 43 values.

This is exactly the architecture from [01_yolo_architecture.md](01_yolo_architecture.md) — nothing extra.

---

## 3. Loss — 3 components

YOLOv8 total loss = **box loss + class loss + dfl loss**. Each teaches the model something different:

### 3.1 Box loss (CIoU)

Teaches the model the correct **location** of the box. Uses **CIoU** (Complete Intersection over Union) — an upgraded version of IoU:

```
IoU  = (intersection) / (union)                  — overlap of two boxes
CIoU = IoU - ρ(distance) - α(aspect_ratio)       — adds a penalty for center distance + aspect ratio
```

CIoU is better than plain IoU because:
- A predicted box with high overlap but a shifted center is still penalized.
- A box with the wrong aspect ratio (predicting a square when the GT is a long rectangle) is penalized.

**Why not MSE (mean squared error) on coordinates?** MSE penalizes every error equally, but a 2px error at (10,10)→(12,12) is semantically very different from a 2px error at (500,500)→(502,502). IoU/CIoU measures "overlap" — more appropriate for detection.

### 3.2 Class loss (BCE)

Teaches the model the correct **class** of the object. Uses **BCE** (Binary Cross-Entropy) — each class is an independent binary problem:

```
BCE(class_score, true_class) = -(t·log(s) + (1-t)·log(1-s))
```

**Why BCE instead of softmax?** Softmax forces the score sum to 1 → the model thinks "only one class is correct". But an image can have **many objects of many classes** at once. BCE lets each cell independently decide "is this class X or not".

### 3.3 DFL loss (Distribution Focal Loss)

DFL teaches the model to predict **box edges** precisely. Instead of predicting a single edge value directly (e.g. 47px), YOLOv8 predicts a **distribution** over discrete values (e.g. 45, 46, 47, 48, 49 → weights [0.05, 0.15, 0.6, 0.15, 0.05]).

DFL improves box accuracy by ~5-10% over direct prediction, especially for small objects. This is advanced — you do not need to touch it, YOLOv8 handles it by default.

> **Practical note:** You never have to modify the loss function. Ultralytics uses CIoU + BCE + DFL automatically. You only need to know "loss going down = model getting better" when reading the training log.

---

## 4. Backpropagation & Optimizer

### Backpropagation

From the loss, compute the **gradient** (partial derivative) for every weight in the network — telling us "should this weight go up or down, and by how much, to reduce the loss". PyTorch `autograd` does this automatically.

### Optimizer — SGD with momentum (YOLOv8 default)

Ultralytics defaults to **SGD with momentum**:
```
w_new = w_old - lr · gradient + momentum · (w_old - w_prev)
```
- `lr` (learning rate): the step size. Too large → overshoot the optimum. Too small → slow.
- `momentum`: carries inertia from the previous step → helps escape local minima and converge faster.

**SGD vs AdamW — why YOLOv8 uses SGD:**

| Optimizer | Pros | Cons |
|---|---|---|
| **SGD momentum** | Converges to better solutions (generalizes better) | Slower, needs lr tuning |
| AdamW | Faster, less tuning needed | Prone to overfit, worse final solution |

Detection needs to generalize (run on new images, different lighting) → SGD is preferred. **Do not switch to AdamW** unless you have a specific reason.

### Learning rate schedule — cos_lr=True

The project uses `cos_lr=True` — **cosine annealing**:

```
lr goes from lr0 down to near 0 following a cosine curve
  │\\\\\\_______
  │  \         \____
  │   \              \____
  │    \                   \____ ~0
  └────────────────────────────→ epoch
```

**Why decay the lr over time?**
- Early in training: large lr → explore quickly, escape local minima.
- Late in training: small lr → fine-tune, "lock in" the precise optimum.

Cosine beats step decay (stepping down in plateaus) because it is smooth — no sudden loss jumps when lr drops. The project sets `cos_lr=True` — best practice for detection.

---

## 5. Transfer learning — why we load `yolov8n.pt`

Training YOLOv8n from scratch on a 39-class citrus dataset (~thousands of images) would **not have enough data** → overfitting or no convergence. Instead:

1. **Pretrain on COCO** (80 classes, 330k images) → the model already "knows how to detect objects" in general (edges, textures, generic shapes).
2. **Fine-tune** on the citrus dataset → only relearn "which class" + the specific features of citrus diseases.

This is transfer learning — reuse general knowledge, then specialize for the narrow task. Notebook cell-7:
```python
model = YOLO("yolov8n.pt")  # auto-downloads pretrained (COCO)
```

**Freeze or not?** Ultralytics fine-tunes **all** layers by default. You can set `freeze=10` (freeze the first 10 backbone layers) to:
- Train faster (fewer gradients to compute).
- Preserve general features (edges, corners) — avoid "forgetting" them.

But with a small dataset + classes very different from COCO (citrus diseases don't look like people/cars), fine-tuning everything usually works better. The project does not freeze → default behavior.

> **Trade-off:** Freeze = faster training, preserves general features, but may be less accurate on unusual classes. No-freeze = slower, but the model fully adapts. With 50 epochs on a T4, no-freeze fits comfortably.

---

## 6. Hyperparameters — why these specific numbers

Notebook cell-8:
```python
EPOCHS = 50
IMGSZ = 640
BATCH = 16
PATIENCE = 15
```

### EPOCHS = 50

YOLOv8n is small and converges fast. 50 epochs is enough for a dataset of a few thousand images. 100+ epochs usually **does not improve** further and only wastes time + risks overfitting.

On free Colab/Kaggle (limited to ~9-12h sessions), 50 epochs of YOLOv8n on a T4 takes ~1-2h → safe, no cutoff.

### BATCH = 16

Batch size = number of images processed at once. Trade-off:
- **Large** (32, 64): more stable gradient (averaged over more images), faster training. But uses more VRAM — a T4 with 15GB may not handle batch 64 + imgsz 640.
- **Small** (4, 8): less VRAM, but noisier gradient, slower convergence.

16 is the sweet spot for T4 + YOLOv8n + 640: fills VRAM just enough to utilize the GPU, gradient is stable enough. Weaker GPU → drop to 8. Stronger GPU (V100) → go to 32.

### IMGSZ = 640

The training image size. **Must match** the ONNX export and the K230 `model_input_size`. Why 640:
- The YOLO standard — well tested.
- Large enough that small objects (diseased leaves) are still visible.
- Small enough that the K230 can handle it realtime (~10-20 FPS).

Switching 640 → 800 (more detail) improves training, but the K230 cannot keep up. Switching → 320 (faster) loses small-object detail → diseases become hard to detect.

### PATIENCE = 15

**Early stopping**: if val loss does not improve for 15 consecutive epochs → stop early. Reasons:
- Saves GPU quota (Kaggle gives 30h/week).
- Prevents overfitting (extra epochs that don't improve val = the model is starting to memorize train rather than learn patterns).

Typically training stops around epoch 30-45, not running the full 50.

> **Practical:** If training ends and loss is still decreasing steadily (no early stop) → you could increase EPOCHS. If it stops very early (< 20) → patience is too small or lr is wrong.

---

## 7. Data augmentation — enriching the data

The model sees more image variations → generalizes better. Ultralytics applies these automatically:

| Augmentation | Default | Effect |
|---|---|---|
| **mosaic** | 1.0 | Stitches 4 images into 1 → the model sees objects in different contexts and sizes |
| **mixup** | 0.0 | Blends 2 images (the project leaves this at 0 — no blending) |
| **fliplr** | 0.5 | Horizontal flip on 50% → the model does not bias toward "left side" |
| **flipud** | 0.0 | Vertical flip — **set to 0 for drones** |
| **hsv_h/s/v** | 0.015/0.7/0.4 | Slight hue/saturation/value changes |
| **degrees** | 0.0 | Rotation (the project leaves this at 0) |
| **scale** | 0.5 | Zoom in/out ±50% |
| **translate** | 0.1 | Shift ±10% |

### Why flipud=0 for a drone?

A drone shoots from above → the image already "looks down". A vertical flip = turning the image upside down → an insect appears upside down → that does not happen in reality → the model learns nonsense. Keep `flipud=0`.

For a horizontal handheld camera, `flipud=0.0` is usually also correct.

### Mosaic — the most important augmentation

4 images stitched:
```
┌──────┬──────┐
│ img1 │ img2 │
├──────┼──────┤
│ img3 │ img4 │
└──────┴──────┘
```
Each quadrant has an object in a different position and size → the model learns "an object can be anywhere, at any scale". Especially helpful for **small objects** (early citrus disease) — when stitched, a small object in img1 occupies an even smaller region → the model learns to detect very small objects.

> **Mosaic turns off at the end of training:** Ultralytics disables mosaic for the last 10 epochs by default (`close_mosaic=10`). Reason: mosaic creates "unreal" images → the final epochs need to fine-tune on real images to lock in the optimum. Do not disable it entirely — leave the default.

---

## 8. Metrics — reading the evaluation results

After training, notebook cell-10:
```python
metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### Precision (P) — "of the boxes I predicted, how many are correct?"

```
P = TP / (TP + FP)
```
- TP (true positive): correct box (IoU > 0.5, correct class)
- FP (false positive): wrong box (no object but the model said there is one)

High P = the model makes few false claims. But it may miss objects (because it is cautious).

### Recall (R) — "of the real objects, how many did the model catch?"

```
R = TP / (TP + FN)
```
- FN (false negative): a real object the model missed

High R = few misses. But it may produce noise (because it is lenient).

**The P vs R trade-off:** Raising the confidence threshold → P goes up (only confident predictions), R goes down (misses weak objects). Lowering the threshold → the opposite.

### mAP50 — mean Average Precision @ IoU 0.5

The average AP across all classes, where a box counts as "correct" if IoU ≥ 0.5. High mAP50 = the model detects objects (location is roughly right).

### mAP50-95 — mean AP @ IoU 0.5 to 0.95

The average AP across many IoU thresholds (0.5, 0.55, 0.6, ..., 0.95). Stricter — requires boxes to be **very precise** to count as correct. mAP50-95 is always lower than mAP50.

### Which metric to trust?

| Metric | When it matters |
|---|---|
| **mAP50** | "Does it detect at all?" — e.g. a drone alerting "there is disease" |
| **mAP50-95** | "Are the boxes pixel-perfect?" — e.g. measuring lesion size |
| **Recall** | Dangerous disease, missing it = severe consequences → prioritize high Recall |
| **Precision** | Too many false alarms = annoying → prioritize high Precision |

**For a citrus-scanning drone:** mAP50 + Recall matter most — you want to detect every diseased leaf, a slightly inaccurate box is fine.

> **Watch out for class imbalance:** the dataset has 39 classes, some rare (e.g. `spring_weevil` has few images) → that class's AP will be low. mAP is an average → common classes can "pull it up". Always look at AP **per class** (`metrics.box.maps`) to see which classes are weak.

---

## 9. Checkpoints & auto-backup — protecting your work

### best.pt vs last.pt

- **best.pt**: the weights with the best val loss/mAP across all training epochs.
- **last.pt**: the weights from the final epoch (may be worse than best if overfitting late).

Deploy uses **best.pt**, never last.pt. (This is why the project only keeps `best.pt` and deleted `last.pt`.)

### The auto-backup callback

Notebook cell-8 has a critical section:
```python
from ultralytics.utils import callbacks

def _backup(trainer):
    try:
        src = os.path.join(trainer.save_dir, "weights", "best.pt")
        shutil.copy(src, os.path.join(OUT_DIR, "best_checkpoint.pt"))
        print(f"  [backup epoch {trainer.epoch}] -> {OUT_DIR}", flush=True)
    except Exception as e:
        print("  [backup fail]", e, flush=True)

callbacks.default_callbacks["on_fit_epoch_end"].append(_backup)
```

**Why this is needed:** Kaggle/Colab sessions can be **interrupted mid-training** (quota exhausted, network error, server reboot). If training reaches epoch 40/50 and then crashes → everything is lost if `best.pt` only lives in `/kaggle/working/runs/...` (not persistent across sessions).

This callback copies `best.pt` to `/kaggle/working/drone_yolo/` **every epoch** — this is Kaggle's **Output folder**, **downloadable at any time** even while the session is still running or just crashed.

> **Real-world lesson:** This project's kernel fell into an ERROR state after cell-11 crashed. But because of the auto-backup, `best_checkpoint.pt` was still on the server → downloadable via the web UI → the entire training effort was saved. **Without this callback, all training work would have been lost.**

### The `on_fit_epoch_end` callback

Ultralytics supports many callback hooks:
- `on_train_start`, `on_train_end`
- `on_epoch_start`, `on_fit_epoch_end` ← used here
- `on_batch_start`, `on_batch_end`

`on_fit_epoch_end` runs **after the epoch is complete + val is done + best.pt has been updated** → the right moment to back up.

---

## 10. Reading the training log

While training, ultralytics prints a table per epoch:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
  1/50   3.2G      1.834      2.941      1.234      142         640
  ...
```
- `box_loss` + `cls_loss` + `dfl_loss`: the 3 losses (section 3) — must decrease over time.
- `GPU_mem`: VRAM in use — if it exceeds 15G (T4) → reduce batch.
- `Instances`: number of objects in the batch.

At the end of each epoch:
```
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                  all        xxx       xxxx      0.xxx      0.xxx      0.xxx      0.xxx
```
- `all`: aggregate across all classes.
- mAP50, mAP50-95 should **increase over time**; small fluctuations are fine.

---

## 11. Summary

| Concept | This project |
|---|---|
| Architecture | YOLOv8n (3M params, 8.1 GFLOPs) |
| Pretrain | COCO 80 classes → fine-tune |
| Optimizer | SGD with momentum (default) |
| LR schedule | Cosine annealing (`cos_lr=True`) |
| Epochs | 50 (patience 15, usually stops ~30-45) |
| Batch | 16 |
| Imgsz | 640 (matches export + K230) |
| Augmentation | mosaic, fliplr 0.5, flipud 0 (drone) |
| Backup | `on_fit_epoch_end` copies best.pt → Output folder |
| Metric to trust | mAP50 + Recall (drone needs to detect everything) |

**Related files:**
- [notebooks/train_yolov8n_drone_kaggle.ipynb](../../notebooks/train_yolov8n_drone_kaggle.ipynb) — cell-8 config + backup, cell-10 metrics, cell-13 export
- [data/citrus/data.yaml](../../data/citrus/data.yaml) — 39 classes + dataset paths

**Next:** [03_data_prep.md](03_data_prep.md) — preparing the dataset, YOLO label format, train/val/test split.
