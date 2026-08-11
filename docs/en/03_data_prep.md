# 03 — Preparing the Dataset for YOLOv8

> Guide to the dataset lifecycle: collect → annotate → export to YOLO format → `data.yaml`. This project uses a Roboflow dataset of 39 citrus classes. References [data/citrus/data.yaml](../../data/citrus/data.yaml).

## 1. Dataset lifecycle — overview

```
1. Collect images       — photograph diseased/healthy citrus leaves at many angles, lighting
2. Annotate             — draw boxes around objects, assign classes
3. Export YOLOv8        — convert to .txt label format + data.yaml
4. Train/val/test split — divide images into 3 sets
5. (optional) Augment   — create image variations to enrich the data
6. Feed to training      — data.yaml points to the paths
```

This project uses **Roboflow** for steps 2-5 (annotate + export + split + augment). You only need to download it in the notebook to start training.

---

## 2. The YOLOv8 format — directory structure

YOLOv8 expects a dataset directory structured like:
```
citrus-disease-detection-1/
├── data.yaml              # config: classes + paths
├── train/
│   ├── images/            # training images (1.jpg, 2.jpg, ...)
│   └── labels/            # training labels (1.txt, 2.txt, ...)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Rule:** Each image `images/xxx.jpg` has one label file `labels/xxx.txt` with the same name (only the extension differs). No .txt = the image has no objects (a background image — also useful).

### data.yaml

The dataset config file. This project has [data/citrus/data.yaml](../../data/citrus/data.yaml):
```yaml
names:                        # list of classes, IN ORDER
- beneficial_insect
- black_aphid
- ...                        # (39 classes total)
- wasp
nc: 39                       # number of classes
train: ../train/images       # path to training images
val: ../valid/images         # path to validation images
test: ../test/images         # path to test images
```

**Important — class order:** `names` is a list; the first class has `id=0`, the second `id=1`, and so on. The model outputs indices according to these ids. If you reorder `names` → a trained model's classes shift → results are completely wrong.

> **Verified:** `k230_yolov8_det.py` has `labels = [...]` — it must match the `names` order in data.yaml **exactly**. This project has synced all 39 classes.

---

## 3. The YOLO label format — 1 line per object

Each `.txt` file (e.g. `citrus_canker_0102.txt`):
```
7 0.4523 0.5612 0.1234 0.0987
7 0.7890 0.2345 0.0567 0.0432
11 0.1234 0.5678 0.2345 0.1890
```

Each line = one object, 5 values:

| Position | Name | Meaning |
|---|---|---|
| 1 | `class_id` | Index in `names` (e.g. `7` = `citrus_canker`) |
| 2 | `cx` | Box center X — **normalized 0-1** (divided by image width) |
| 3 | `cy` | Box center Y — **normalized 0-1** (divided by image height) |
| 4 | `w` | Box width — **normalized 0-1** (divided by image width) |
| 5 | `h` | Box height — **normalized 0-1** (divided by image height) |

### Why normalized?

- Independent of image size (640, 800, 1024) → boxes are correct at any imgsz.
- Easy to augment (flip, scale) — just multiply/shift coefficients.
- Values always in [0,1] → stable gradients.

### Decode example

Image 640×480, box `7 0.5 0.5 0.25 0.4`:
- Center: (0.5×640, 0.5×480) = (320, 240)
- Size: (0.25×640, 0.4×480) = (160, 192)
- Box in pixels: x1=240, y1=144, x2=400, y2=336
- Class: `citrus_canker` (id 7)

---

## 4. Train / Val / Test split — why 3 sets

| Set | Purpose | Used when |
|---|---|---|
| **train** | The model **learns** from here (backprop) | Throughout training |
| **val** | Evaluate each epoch, choose `best.pt` | Throughout training (after each epoch) |
| **test** | Final evaluation, **only once** | After training is done |

### Why val is separate from train?

If you evaluate on the training data itself → the model may **overfit** (memorize) and still "score high". Val is data the model **has not seen** → a true measurement.

### Why test is separate from val?

During training, you implicitly "tune" based on val (choosing best.pt, adjusting patience, comparing configs). Val becomes "partially learned" → test (completely unseen) is the only objective measure.

This project's Roboflow split is 70/20/10 (train/val/test) by default — standard.

> **Note:** **Never** select best.pt based on test. Test is used **once, at the end**. Selecting based on test = fooling yourself (overfitting to test).

---

## 5. The 39-class citrus dataset — overview

This project trains on the "Citrus Disease Detection" dataset (Roboflow). 39 classes in 3 groups:

**Diseases — ~15 classes:**
citrus_anthracnose, citrus_black_spot, citrus_brown_spot, citrus_canker, citrus_exocortis, citrus_greasy_spot, citrus_huanglongbing, citrus_leprosis, citrus_melanose, citrus_powdery_mildew, citrus_rot, citrus_rust_mite, citrus_scab, citrus_sooty_mold, other_disease

**Pests — ~15 classes:**
black_aphid, brown_banded_tortrix, citrus_aphid, citrus_fruit_fly, citrus_leafminer, citrus_longhorned_beetle, citrus_psyllid, citrus_red_mite, citrus_swallowtail, citrus_thrips, citrus_whitefly, other_pest, other_scale_insect, other_slug_moth, red_wax_scale, spring_weevil, thripidae

**Beneficial / healthy — ~5 classes:**
beneficial_insect, healthy_leaf, honeybee, lacewing, ladybug, spider, wasp

### Why distinguish pest vs beneficial?

The drone needs to know "there is a pest" (needs treatment) vs "there is a beneficial insect" (do not treat). `ladybug`, `lacewing`, `honeybee` eat aphids → beneficial, do not spray. Correct distinction → reduces pesticide waste, protects the ecosystem.

---

## 6. Class imbalance — the problem + solutions

The 39-class dataset is **not balanced**:
- Common classes (`citrus_canker`, `healthy_leaf`): hundreds of images.
- Rare classes (`spring_weevil`, `citrus_longhorned_beetle`): a few dozen images.

**Consequence:** the model biases toward common classes → reports `citrus_canker` for every similar lesion → low Recall on rare classes.

**Automatic solutions (Ultralytics):**

1. **Mosaic augmentation** (section 7 of doc 02): stitches 4 images → a rare class can appear alongside common ones → more balanced gradients.
2. **Class weighting** (implicit): Ultralytics has a mechanism to balance loss by class frequency.

**Manual solutions (if needed):**

- **Oversampling**: duplicate images of rare classes (simple, but may overfit).
- **SMOTE-like**: augment rare classes more aggressively (rotate, crop).
- **Collect more data**: photograph more rare-class samples (best, but labor-intensive).

> **Check:** After training, look at `metrics.box.maps` (per-class AP). Any class with AP < 0.3 → weak. You can augment more, or accept it (if that class is not critical for your use case).

---

## 7. Roboflow augmentation vs Ultralytics augmentation

Roboflow has a **Generate Version** step — lets you choose preprocessing + augmentation:
- Resize, auto-orient (preprocess)
- Rotate, flip, brightness, noise (augment)

**Why this project augments in Ultralytics (notebook), not Roboflow?**

Roboflow augments **statically** — generates new images and stores them in the dataset → the dataset grows 3-5× → slow download, more disk usage.

Ultralytics augments **dynamically** — applies random transforms each epoch on the original images → the dataset does not grow, but the model still sees new variations every epoch. More flexible and more effective.

**Best practice:** Use Roboflow only for preprocessing (standardize resize) + splitting. Leave augmentation to Ultralytics.

---

## 8. cache=True — speeding up training

Notebook cell-8: `cache=True`. Two modes:

| Mode | How | Pros | Cons |
|---|---|---|---|
| `cache=True` (RAM) | Load all images into RAM once | Fastest (no disk reads) | Uses RAM — large datasets may exceed RAM |
| `cache="disk"` | Save compressed images to a disk cache | Less RAM | Slower than RAM |
| `cache=False` | Read from disk every batch | No RAM use | Slowest |

This project's dataset is a few thousand images, a few hundred MB → fits Kaggle/Colab RAM (12-16GB) → `cache=True` is safe and trains ~20-30% faster.

> **Note:** If you hit OOM (out of RAM) during training → switch to `cache="disk"` or `cache=False`. Reducing batch also helps.

---

## 9. Collecting images — best practices for drones

If you later shoot your own dataset:

### Diverse conditions
- **Lighting**: harsh sun, shade, backlit, late afternoon.
- **Angle**: from above (drone flying), oblique, horizontal.
- **Distance**: close (detail), far (small objects).
- **Background**: single leaf, dense canopy, branches遮挡.

### Image quality
- Sufficient light, no motion blur.
- Resolution ≥ 640×640 at the object region (so detail survives after resize).
- No color filters / over-saturation (loses real features).

### Quantity
- Common classes: 200+ images.
- Rare classes: minimum 50-100 images (below 50 → very low AP).
- Background images (no objects): 10-20% of total → reduces false positives.

> **Drone-specific:** Shoot at the altitude the drone will actually fly (e.g. 2-5m). Do not shoot close-ups and hope the model generalizes to high-altitude images — the angle and object size are completely different.

---

## 10. Summary

| Concept | This project |
|---|---|
| Dataset source | Roboflow Citrus Disease Detection v1 |
| Format | YOLOv8 (.txt + data.yaml) |
| Split | train/valid/test (70/20/10 Roboflow) |
| Classes | 39 (diseases + pests + beneficial + healthy) |
| Augmentation | Ultralytics dynamic (mosaic, fliplr, hsv, scale) |
| Cache | RAM (`cache=True`) |
| data.yaml | `data/citrus/data.yaml` (metadata only, images on Roboflow/Kaggle) |

**Related files:**
- [data/citrus/data.yaml](../../data/citrus/data.yaml) — 39 class names + config
- [data/citrus/README.dataset.txt](../../data/citrus/README.dataset.txt) — Roboflow description
- [notebooks/train_yolov8n_drone_kaggle.ipynb](../../notebooks/train_yolov8n_drone_kaggle.ipynb) — cell-4 downloads Roboflow, cell-5 finds data.yaml

**Next:** [04_export_deploy.md](04_export_deploy.md) — export to ONNX, convert to kmodel, deploy on the K230.
