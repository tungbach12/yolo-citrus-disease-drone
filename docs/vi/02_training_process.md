# 02 — Quá trình Training YOLOv8

> Tài liệu tự học về **training loop** — mạng học gì, tối ưu thế nào, dừng khi nào. Project train 50 epochs YOLOv8n trên Kaggle GPU T4. Tham chiếu [notebooks/train_yolov8n_drone_kaggle.ipynb](../../notebooks/train_yolov8n_drone_kaggle.ipynb) cell-8.

## 1. Training loop — vòng lặp cốt lõi

Mỗi **epoch** (vòng) train, mô hình duyệt **toàn bộ** dataset một lần. Mỗi **batch** (16 ảnh) bên trong epoch:

```
for epoch in range(50):              # EPOCHS = 50
    for batch in dataset:            # mỗi batch 16 ảnh
        1. Forward pass      — mô hình dự đoán box
        2. Tính loss         — so sánh dự đoán vs ground truth
        3. Backpropagation   — tính gradient (đạo hàm loss theo trọng số)
        4. Optimizer step    — cập nhật trọng số giảm loss
    # hết epoch → đánh giá trên val → maybe save best.pt
```

Sau nhiều epoch, loss giảm dần, dự đoán gần ground truth → mô hình "học được".

---

## 2. Forward pass

Batch ảnh `(16, 3, 640, 640)` đi qua mạng (backbone → neck → head) → ra output `(16, 43, 8400)` = dự đoán cho 16 ảnh, mỗi ảnh 8400 cell × 43 giá trị.

Đây chính là kiến trúc ở [01_yolo_architecture.md](01_yolo_architecture.md) — không có gì thêm.

---

## 3. Loss — 3 thành phần

YOLOv8 loss tổng = **box loss + class loss + dfl loss**. Mỗi thành phần dạy mô hình một thứ:

### 3.1 Box loss (CIoU)

Dạy mô hình **vị trí** box đúng. Dùng **CIoU** (Complete Intersection over Union) — phiên bản nâng cấp của IoU:

```
IoU = (giao) / (hợp)                  — overlap 2 box
CIoU = IoU - ρ(distance) - α(aspect)  — thêm penalty khoảng cách tâm + tỷ lệ cạnh
```

CIoU tốt hơn IoU thường vì:
- Box dự đoán dù overlap cao nhưng tâm lệch → vẫn bị phạt.
- Box sai tỷ lệ (dự đoán box vuông, GT chữ nhật dài) → bị phạt.

**Tại sao không dùng MSE (mean squared error) cho tọa độ?** MSE phạt đều mọi sai lệch, nhưng box (10,10)→(12,12) sai 2px rất khác (500,500)→(502,502) sai 2px về ngữ nghĩa. IoU/CIoU đo "overlap" — hợp lý hơn cho detection.

### 3.2 Class loss (BCE)

Dạy mô hình **loại** vật thể đúng. Dùng **BCE** (Binary Cross-Entropy) — mỗi class là bài toán binary độc lập:

```
BCE(class_score, true_class) = -(t·log(s) + (1-t)·log(1-s))
```

**Vì sao BCE không phải softmax?** Softroidx ép tổng score = 1 → mô hình nghĩ "chỉ 1 class đúng". Nhưng ảnh có thể có **nhiều vật thể nhiều class** cùng lúc. BCE cho phép mỗi cell tự quyết "có phải class X không" độc lập.

### 3.3 DFL loss (Distribution Focal Loss)

DFL dạy mô hình **cạnh box** chính xác. Thay vì dự đoán trực tiếp giá trị cạnh (vd 47px), YOLOv8 dự đoán **phân phối** khả năng trên các giá trị rời rạc (vd: 45, 46, 47, 48, 49 → trọng số [0.05, 0.15, 0.6, 0.15, 0.05]).

DFL giúp box chính xác hơn ~5-10% so với dự đoán trực tiếp, đặc biệt vật thể nhỏ. Đây là phần kỹ thuật nâng cao — bạn không cần can thiệp, mặc định YOLOv8 đã xử lý.

> **Thực hành:** Bạn không phải sửa loss function bao giờ. Ultralytics tự dùng CIoU + BCE + DFL. Chỉ cần biết "loss giảm = mô hình tốt hơn" khi đọc log train.

---

## 4. Backpropagation & Optimizer

### Backpropagation

Từ loss, tính **gradient** (đạo hàm riêng) cho từng trọng số trong mạng — cho biết "trọng số này nên tăng hay giảm, bao nhiêu, để loss giảm". PyTorch `autograd` làm tự động.

### Optimizer — SGD momentum (mặc định YOLOv8)

Ultralytics mặc định **SGD với momentum**:
```
w_new = w_old - lr · gradient + momentum · (w_old - w_prev)
```
- `lr` (learning rate): bước nhảy. Quá lớn → nhảy qua điểm tốt. Quá nhỏ → chậm.
- `momentum`: đà từ bước trước → vượt qua local minimum, hội tụ nhanh hơn.

**SGD vs AdamW — vì sao YOLOv8 dùng SGD?**

| Optimizer | Ưu | Nhược |
|---|---|---|
| **SGD momentum** | Hội tụ đến nghiệm tốt hơn (generalize tốt) | Chậm hơn, cần tuning lr |
| AdamW | Nhanh, ít cần tuning | Dễ overfit, nghiệm kém "sạch" |

Detection cần generalize (chạy trên ảnh mới, điều kiện sáng khác) → SGD được ưu tiên. **Đừng đổi sang AdamW** trừ khi có lý do cụ thể.

### Learning rate schedule — cos_lr=True

Project dùng `cos_lr=True` — **cosine annealing**:

```
lr đi từ lr0 xuống gần 0 theo đường cosine
  │\\\\\\_______
  │  \         \____
  │   \              \____
  │    \                   \____ ~0
  └────────────────────────────→ epoch
```

**Vì sao giảm dần lr?**
- Đầu train: lr lớn → khám phá nhanh, vượt local minimum.
- Cuối train: lr nhỏ → tinh chỉnh, "chốt" nghiệm chính xác.

Cosine优于 step decay (giảm theo bậc) vì mượt, không bị "đứt" loss khi lr đổi đột ngột. Project để `cos_lr=True` — best practice cho detection.

---

## 5. Transfer learning — vì sao load `yolov8n.pt`

YOLOv8n train từ scratch trên dataset 39 class citrus (~hàng nghìn ảnh) sẽ **không đủ data** → overfit hoặc không hội tụ. Thay vào đó:

1. **Pretrain trên COCO** (80 class, 330k ảnh) → mô hình đã biết "phát hiện vật thể" nói chung (cạnh, texture, hình dạng chung).
2. **Fine-tune** trên citrus dataset → chỉ học lại "class nào" + đặc trưng bệnh cam cụ thể.

Đó là transfer learning — dùng kiến thức tổng quát có sẵn, tinh chỉnh cho bài toán hẹp. Cell-7 notebook:
```python
model = YOLO("yolov8n.pt")  # tự tải pretrained (COCO)
```

**Freeze hay không?** Ultralytics fine-tune **toàn bộ** layer mặc định. Có thể `freeze=10` (đóng băng 10 layer đầu backbone) để:
- Train nhanh hơn (ít gradient).
- Giữ đặc trưng tổng quát (cạnh, góc) — không bị "quên".

Nhưng với dataset nhỏ + class rất khác COCO (bệnh cam không giống người/xe), fine-tune toàn bộ thường tốt hơn. Project không freeze → mặc định.

> **Trade-off:** Freeze = train nhanh, giữ feature chung, nhưng có thể kém chính xác trên class lạ. No-freeze = chậm hơn, nhưng mô hình thích nghi hoàn toàn. Với 50 epochs trên T4, không freeze vừa vặn.

---

## 6. Siêu tham số (hyperparameters) — vì sao các con số đó

Cell-8 notebook:
```python
EPOCHS = 50
IMGSZ = 640
BATCH = 16
PATIENCE = 15
```

### EPOCHS = 50

YOLOv8n nhỏ, hội tụ nhanh. 50 epochs đủ cho dataset vài nghìn ảnh. 100+ epochs thường **không cải thiện** thêm mà chỉ tốn thời gian + nguy cơ overfit.

Trên Colab/Kaggle miễn phí (giới hạn ~9-12h session), 50 epochs YOLOv8n trên T4 mất ~1-2h → an toàn, không bị cut.

### BATCH = 16

Batch size = số ảnh xử lý đồng thời. Trade-off:
- **Lớn** (32, 64): gradient ổn hơn (trung bình nhiều ảnh), train nhanh. Nhưng tốn VRAM — T4 15GB có thể không đủ với batch 64 + imgsz 640.
- **Nhỏ** (4, 8): ít VRAM, nhưng gradient nhiễu, chậm hội tụ.

16 là sweet spot cho T4 + YOLOv8n + 640: vừa đầy VRAM để tận dụng GPU, vừa gradient đủ ổn. Nếu GPU yếu hơn → giảm 8. GPU mạnh hơn (V100) → tăng 32.

### IMGSZ = 640

Kích thước ảnh train. **Phải khớp** với export ONNX và `model_input_size` K230. Lý do 640:
- Tiêu chuẩn YOLO — được test kỹ.
- Đủ lớn để vật thể nhỏ (lá bệnh) còn thấy.
- Đủ nhỏ để K230 xử lý được realtime (~10-20 FPS).

Đổi 640 → 800 (chi tiết hơn) thì train tốt hơn, nhưng K230 không chạy kịp. Đổi → 320 (nhanh hơn) thì mất chi tiết nhỏ → bệnh khó phát hiện.

### PATIENCE = 15

**Early stopping**: nếu val loss không cải thiện sau 15 epoch liên tiếp → dừng sớm. Lý do:
- Tiết kiệm GPU quota (Kaggle 30h/tuần).
- Tránh overfit (epoch dư không cải thiện val = bắt đầu nhớ train thay vì học quy luật).

Thường train dừng quanh epoch 30-45, không chạy hết 50.

> **Thực hành:** Nếu train kết thúc vẫn giảm loss đều (chưa dừng sớm) → có thể tăng EPOCHS. Nếu dừng sớm sớm quá (< 20) → patience quá nhỏ hoặc lr sai.

---

## 7. Data augmentation — làm giàu dữ liệu

Mô hình thấy nhiều biến thể ảnh hơn → generalize tốt hơn. Ultralytics tự áp dụng:

| Augmentation | Mặc định | Tác dụng |
|---|---|---|
| **mosaic** | 1.0 | Ghép 4 ảnh thành 1 → mô hình thấy vật thể ở context khác, kích thước khác |
| **mixup** | 0.0 | Trộn 2 ảnh (project để 0 — không trộn) |
| **fliplr** | 0.5 | Lật ngang 50% → mô hình không thiên "bên trái" |
| **flipud** | 0.0 | Lật dọc — **để 0 cho drone** |
| **hsv_h/s/v** | 0.015/0.7/0.4 | Đổi sắc độ/sáng/độ tương phản nhẹ |
| **degrees** | 0.0 | Xoay (project để 0) |
| **scale** | 0.5 | Zoom in/out ±50% |
| **translate** | 0.1 | Dịch chuyển ±10% |

### Vì sao flipud=0 cho drone?

Drone chụp từ trên cao → ảnh đã "nhìn xuống". Lật dọc = lật ảnh ngược đầu → con côn trùng lộn ngược → không có trong thực tế → mô hình học nhậy. Giữ `flipud=0`.

Ảnh camera ngang (người cầm điện thoại) thì `flipud=0.0` cũng thường đúng.

### Mosaic — augmentation quan trọng nhất

4 ảnh ghép:
```
┌──────┬──────┐
│ img1 │ img2 │
├──────┼──────┤
│ img3 │ img4 │
└──────┴──────┘
```
Mỗi ô 1 vật thể ở vị trí khác, kích thước khác → mô hình học "vật thể có thể ở bất kỳ đâu, to/nhỏ thế nào". Đặc biệt giúp **vật thể nhỏ** (bệnh cam sớm) — khi ghép, vật thể nhỏ trong img1 chiếm vùng nhỏ hơn cả ô → mô hình học phát hiện vật thể rất nhỏ.

> **Tắt mosaic cuối train:** Ultralytics tự tắt mosaic 10 epoch cuối (giá trị `close_mosaic=10`). Lý do: mosaic tạo ảnh "không thật" → cuối train cần tinh chỉnh trên ảnh thật để chốt nghiệm. Đừng tắt hẳn — cứ để mặc định.

---

## 8. Metrics — đọc kết quả đánh giá

Sau train, cell-10 notebook:
```python
metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### Precision (P) — "nói có vật thể, đúng bao nhiêu %"

```
P = TP / (TP + FP)
```
- TP (true positive): box đúng (IoU > 0.5, class đúng)
- FP (false positive): box sai (không có vật thể mà mô hình báo có)

P cao = mô hình ít báo sai. Nhưng có thể bỏ sót (vì thận trọng).

### Recall (R) — "vật thể thật, mô hình bắt bao nhiêu %"

```
R = TP / (TP + FN)
```
- FN (false negative): có vật thể mà mô hình bỏ sót

R cao = ít sót. Nhưng có thể báo nhiễu (vì dễ dãi).

**Trade-off P vs R:** Tăng confidence threshold → P tăng (chỉ báo chắc chắn), R giảm (bỏ sót vật thể yếu). Giảm threshold → ngược lại.

### mAP50 — mean Average Precision @ IoU 0.5

Trung bình AP của tất cả class, với box tính "đúng" nếu IoU ≥ 0.5. mAP50 cao = mô hình phát hiện được vật thể (vị trí tương đối đúng).

### mAP50-95 — mean AP @ IoU 0.5 đến 0.95

Trung bình AP qua nhiều ngưỡng IoU (0.5, 0.55, 0.6, ..., 0.95). Khắt khe hơn — đòi hỏi box **rất chính xác** mới tính đúng. mAP50-95 luôn thấp hơn mAP50.

### Báo cáo nào tin?

| Metric | Khi nào quan trọng |
|---|---|
| **mAP50** | Muốn biết "phát hiện được không" — ví dụ drone cảnh báo "có bệnh" |
| **mAP50-95** | Muốn box chính xác tuyệt đối — ví dụ đo kích thước vết bệnh |
| **Recall** | Bệnh nguy hiểm, sót = hậu quả nặng → ưu tiên Recall cao |
| **Precision** | Báo sai nhiều = phiền → ưu tiên Precision cao |

**Cho drone quét cam:** mAP50 + Recall quan trọng hơn — cần phát hiện hết lá bệnh, box sai chút không sao.

> **Lưu ý class imbalance:** dataset 39 class, một số class hiếm (vd `spring_weevil` ít ảnh) → AP class đó thấp. mAP là trung bình → có thể bị class phổ biến "kéo lên". Luôn xem AP **theo từng class** (`metrics.box.maps`) để biết class nào yếu.

---

## 9. Checkpoint & auto-backup — chống mất công

### best.pt vs last.pt

- **best.pt**: weights có val loss/mAP tốt nhất trong toàn bộ epoch train.
- **last.pt**: weights epoch cuối cùng (có thể kém hơn best nếu overfit cuối).

Deploy dùng **best.pt**, không bao giờ dùng last.pt. (Đó là vì sao project chỉ giữ `best.pt`, đã xóa `last.pt`.)

### Auto-backup callback

Cell-8 notebook có đoạn quan trọng:
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

**Vì sao cần?** Kaggle/Colab session có thể bị **ngắt giữa chừng** (hết quota, lỗi mạng, server reboot). Nếu train đến epoch 40/50 rồi crash → mất hết nếu `best.pt` chỉ nằm trong `/kaggle/working/runs/...` (không persistent qua session).

Callback này **mỗi epoch** copy `best.pt` ra `/kaggle/working/drone_yolo/` — đây là **Output folder** của Kaggle, **download được bất kỳ lúc nào** kể cả khi session còn chạy hoặc vừa crash.

> **Bài học thực tế:** Project này kernel rơi vào ERROR state sau cell-11 crash. Nhưng vì auto-backup, `best_checkpoint.pt` vẫn có trên server → download qua web UI được → cứu được toàn bộ công train. **Không có callback này = mất hết.**

### Callback `on_fit_epoch_end`

Ultralytics hỗ trợ nhiều callback hook:
- `on_train_start`, `on_train_end`
- `on_epoch_start`, `on_fit_epoch_end` ← dùng ở đây
- `on_batch_start`, `on_batch_end`

`on_fit_epoch_end` chạy **sau khi epoch hoàn tất + val xong + best.pt đã update** → thời điểm đúng để backup.

---

## 10. Quá trình train thực tế — đọc log

Khi train, ultralytics in bảng mỗi epoch:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
  1/50   3.2G      1.834      2.941      1.234      142         640
  ...
```
- `box_loss` + `cls_loss` + `dfl_loss`: 3 loss (mục 3) — phải giảm dần.
- `GPU_mem`: VRAM đang dùng — nếu quá 15G (T4) → giảm batch.
- `Instances`: số vật thể trong batch.

Cuối mỗi epoch:
```
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                  all        xxx       xxxx      0.xxx      0.xxx      0.xxx      0.xxx
```
- `all`: tổng hợp toàn class.
- mAP50, mAP50-95 phải **tăng dần**, dao động nhỏ OK.

---

## 11. Tổng kết

| Khái niệm | Project dùng |
|---|---|
| Kiến trúc | YOLOv8n (3M params, 8.1 GFLOPs) |
| Pretrain | COCO 80 class → fine-tune |
| Optimizer | SGD momentum (mặc định) |
| LR schedule | Cosine annealing (`cos_lr=True`) |
| Epochs | 50 (patience 15, thường dừng ~30-45) |
| Batch | 16 |
| Imgsz | 640 (khớp export + K230) |
| Augmentation | mosaic, fliplr 0.5, flipud 0 (drone) |
| Backup | `on_fit_epoch_end` copy best.pt → Output folder |
| Metric tin | mAP50 + Recall (drone cần phát hiện hết) |

**File liên quan:**
- [notebooks/train_yolov8n_drone_kaggle.ipynb](../../notebooks/train_yolov8n_drone_kaggle.ipynb) — cell-8 config + backup, cell-10 metrics, cell-13 export
- [data/citrus/data.yaml](../../data/citrus/data.yaml) — 39 class + đường dẫn dataset

**Tiếp theo:** [03_data_prep.md](03_data_prep.md) — chuẩn bị dataset, format YOLO label, train/val/test split.
