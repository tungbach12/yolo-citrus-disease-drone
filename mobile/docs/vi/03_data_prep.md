# 03 — Chuẩn bị Dataset cho YOLOv8

> Tài liệu về vòng đời dataset: thu thập → gán nhãn → export YOLO format → `data.yaml`. Project dùng Roboflow dataset 39 class citrus. Tham chiếu [data/citrus/data.yaml](../../data/citrus/data.yaml).

## 1. Vòng đời dataset — tổng quan

```
1. Thu thập ảnh      — chụp lá cam bệnh/tốt từ nhiều góc, ánh sáng
2. Gán nhãn (annotate) — vẽ box quanh vật thể, gán class
3. Export YOLOv8     — chuyển sang format .txt + data.yaml
4. Train/val/test split — chia ảnh thành 3 tập
5. (tùy chọn) Augment — biến thể ảnh làm giàu data
6. Đưa vào train      — data.yaml trỏ đường dẫn
```

Project dùng **Roboflow** làm bước 2-5 (annotate + export + split + augment). Chỉ cần download vào notebook là train được.

---

## 2. Format YOLOv8 — cấu trúc thư mục

YOLOv8 expect thư mục dataset có cấu trúc:
```
citrus-disease-detection-1/
├── data.yaml              # cấu hình: class + đường dẫn
├── train/
│   ├── images/            # ảnh train (1.jpg, 2.jpg, ...)
│   └── labels/            # nhãn train (1.txt, 2.txt, ...)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Quy tắc:** Mỗi ảnh `images/xxx.jpg` có 1 file nhãn `labels/xxx.txt` cùng tên (chỉ đuôi khác). Không có .txt = ảnh không có vật thể (background image — cũng hữu ích).

### data.yaml

File cấu hình dataset. Project có [data/citrus/data.yaml](../../data/citrus/data.yaml):
```yaml
names:                        # danh sách class, ĐÚNG THỨ TỰ
- beneficial_insect
- black_aphid
- ...                        # (39 class tổng cộng)
- wasp
nc: 39                       # số class
train: ../train/images       # đường dẫn ảnh train
val: ../valid/images         # đường dẫn ảnh val
test: ../test/images         # đường dẫn ảnh test
```

**Quan trọng — thứ tự class:** `names` là list, class đầu có `id=0`, class thứ hai `id=1`, ... Mô hình output index theo id này. Nếu đổi thứ tự names → mô hình đã train bị lệch class → kết quả sai hoàn toàn.

> **Kiểm chứng:** `k230_yolov8_det.py` có `labels = [...]` — phải khớp **chính xác** thứ tự `names` trong data.yaml. Project đã đồng bộ 39 class.

---

## 3. Format nhãn YOLO — 1 dòng / 1 vật thể

Mỗi file `.txt` (vd `citrus_canker_0102.txt`):
```
7 0.4523 0.5612 0.1234 0.0987
7 0.7890 0.2345 0.0567 0.0432
11 0.1234 0.5678 0.2345 0.1890
```

Mỗi dòng = 1 vật thể, 5 giá trị:

| Vị trí | Tên | Ý nghĩa |
|---|---|---|
| 1 | `class_id` | Index trong `names` (vd `7` = `citrus_canker`) |
| 2 | `cx` | Tọa độ tâm box X — **normalized 0-1** (chia chiều rộng ảnh) |
| 3 | `cy` | Tọa độ tâm box Y — normalized 0-1 (chia chiều cao ảnh) |
| 4 | `w` | Rộng box — normalized 0-1 (chia chiều rộng ảnh) |
| 5 | `h` | Cao box — normalized 0-1 (chia chiều cao ảnh) |

### Vì sao normalized?

- Không phụ thuộc kích thước ảnh (640, 800, 1024) → box đúng trên mọi imgsz.
- Dễ augment (lật, scale) — chỉ nhân/tịnh tiến hệ số.
- Giá trị luôn trong [0,1] → gradient ổn.

### Ví dụ decode

Ảnh 640×480, box `7 0.5 0.5 0.25 0.4`:
- Tâm: (0.5×640, 0.5×480) = (320, 240)
- Kích thước: (0.25×640, 0.4×480) = (160, 192)
- Box pixel: x1=240, y1=144, x2=400, y2=336
- Class: `citrus_canker` (id 7)

---

## 4. Train / Val / Test split — vì sao 3 tập

| Tập | Mục đích | Dùng khi |
|---|---|---|
| **train** | Mô hình **học** từ đây (backprop) | Suốt train |
| **val** | Đánh giá mỗi epoch, chọn `best.pt` | Suốt train (sau mỗi epoch) |
| **test** | Đánh giá cuối cùng, **chỉ 1 lần** | Sau train xong |

### Vì sao val tách train?

Nếu đánh giá trên chính data train → mô hình có thể **overfit** (học thuộc lòng) mà vẫn "điểm cao". Val là data mô hình **chưa thấy** → đo thật sự.

### Vì sao test tách val?

Suốt train, mình ngầm "tuning" dựa trên val (chọn best.pt, chỉnh patience, so sánh config). Val trở thành "đã ngầm học" phần nào → test (hoàn toàn mới) mới đo khách quan.

Project Roboflow chia 70/20/10 (train/val/test) mặc định — phổ biến.

> **Lưu ý:** **Không bao giờ** chọn best.pt dựa trên test. Test chỉ dùng **1 lần cuối**. Nếu chọn theo test = lừa mình (overfit test).

---

## 5. Dataset citrus 39 class — tổng quan

Project train trên dataset "Citrus Disease Detection" (Roboflow). 39 class chia 3 nhóm:

**Bệnh (disease) — ~15 class:**
citrus_anthracnose, citrus_black_spot, citrus_brown_spot, citrus_canker, citrus_exocortis, citrus_greasy_spot, citrus_huanglongbing, citrus_leprosis, citrus_melanose, citrus_powdery_mildew, citrus_rot, citrus_rust_mite, citrus_scab, citrus_sooty_mold, other_disease

**Dịch hại (pest) — ~15 class:**
black_aphid, brown_banded_tortrix, citrus_aphid, citrus_fruit_fly, citrus_leafminer, citrus_longhorned_beetle, citrus_psyllid, citrus_red_mite, citrus_swallowtail, citrus_thrips, citrus_whitefly, other_pest, other_scale_insect, other_slug_moth, red_wax_scale, spring_weevil, thripidae

**Có ích / khỏe — ~5 class:**
beneficial_insect, healthy_leaf, honeybee, lacewing, ladybug, spider, wasp

### Tại sao phân biệt pest vs beneficial?

Drone cần biết "có sâu bệnh" (cần xử lý) vs "có côn trùng có ích" (không xử lý). `ladybug`, `lacewing`, `honeybee` ăn rệp/aphid → có ích, không diệt. Phân biệt đúng → giảm thuốc trừ sâu lãng phí, bảo vệ hệ sinh thái.

---

## 6. Class imbalance — vấn đề + giải pháp

Dataset 39 class nhưng **không đều**:
- Class phổ biến (`citrus_canker`, `healthy_leaf`): hàng trăm ảnh.
- Class hiếm (`spring_weevil`, `citrus_longhorned_beetle`): vài chục ảnh.

**Hệ quả:** mô hình thiên class phổ biến → báo `citrus_canker` cho mọi vết bệnh tương tự → Recall class hiếm thấp.

**Giải pháp tự động (Ultralytics):**

1. **Mosaic augmentation** (mục 7 doc 02): ghép 4 ảnh → class hiếm có cơ hội xuất hiện cùng class phổ biến → gradient đều hơn.
2. **Class weighting** (ngầm): Ultralytics có cơ chế cân bằng loss theo tần suất class.

**Giải pháp thủ công (nếu cần):**

- **Oversampling**: duplicate ảnh class hiếm (đơn giản, có thể overfit).
- **SMOTE-like**: augment mạnh class hiếm riêng (xoay, crop).
- **Thu thêm data**: chụp thêm ảnh class hiếm (tốt nhất, tốn công).

> **Kiểm tra:** Sau train, xem `metrics.box.maps` (AP từng class). Class nào AP < 0.3 → yếu. Có thể augment thêm hoặc chấp nhận (nếu class đó không quan trọng trong use case).

---

## 7. Augment trên Roboflow vs Ultralytics

Roboflow có bước **Generate Version** — cho chọn preprocessing + augmentation:
- Resize, auto-orient (preprocess)
- Rotate, flip, brightness, noise (augment)

**Vì sao project augment trên Ultralytics (notebook), không phải Roboflow?**

Roboflow augment **tĩnh** — sinh ảnh mới lưu vào dataset → dataset to gấp 3-5 lần → download chậm, tốn dung lượng.

Ultralytics augment **động** — áp dụng ngẫu nhiên mỗi epoch trên ảnh gốc → dataset không to, nhưng mô hình vẫn thấy biến thể mới mỗi epoch. Linh hoạt hơn, hiệu quả hơn.

**Best practice:** Roboflow chỉ preprocess (resize chuẩn) + split. Augment để Ultralytics xử lý.

---

## 8. cache=True — tăng tốc train

Cell-8 notebook: `cache=True`. Có 2 chế độ:

| Chế độ | Cách | Ưu | Nhược |
|---|---|---|---|
| `cache=True` (RAM) | Load toàn bộ ảnh vào RAM lần đầu | Nhanh nhất (không đọc disk) | Tốn RAM — dataset lớn > RAM thì lỗi |
| `cache="disk"` | Lưu ảnh dạng nén vào disk cache | Ít tốn RAM | Chậm hơn RAM |
| `cache=False` | Đọc disk mỗi batch | Không tốn RAM | Chậm nhất |

Project dataset ~hàng nghìn ảnh vài trăm MB → vừa RAM Kaggle/Colab (12-16GB) → `cache=True` an toàn, train nhanh ~20-30%.

> **Lưu ý:** Nếu OOM (hết RAM) giữa train → đổi `cache="disk"` hoặc `cache=False`. Giảm batch cũng giúp.

---

## 9. Thu thập ảnh — best practice cho drone

Nếu sau này tự chụp dataset:

### Đa dạng điều kiện
- **Ánh sáng**: sáng gắt, râm mát, ngược sáng, chiều muộn.
- **Góc chụp**: trên cao (drone bay), nghiêng, ngang.
- **Khoảng cách**: gần (chi tiết), xa (vật thể nhỏ).
- **Nền**: lá đơn, tán dày, có cành che.

### Chất lượng ảnh
- Đủ sáng, không mờ rung.
- Độ phân giải ≥ 640×640 ở vùng vật thể (để sau resize còn chi tiết).
- Không lọc màu/quá saturation (mất đặc trưng thật).

### Số lượng
- Class phổ biến: 200+ ảnh.
- Class hiếm: tối thiểu 50-100 ảnh (dưới 50 → AP rất thấp).
- Background images (ảnh không có vật thể): 10-20% tổng → giảm false positive.

> **Drone cụ thể:** Chụp ở độ cao drone sẽ bay thực tế (vd 2-5m). Đừng chụp cận cảnh rồi hy vọng mô hình generalize sang ảnh bay cao — góc + kích thước vật thể khác hoàn toàn.

---

## 10. Tổng kết

| Khái niệm | Project |
|---|---|
| Nguồn dataset | Roboflow Citrus Disease Detection v1 |
| Format | YOLOv8 (.txt + data.yaml) |
| Split | train/valid/test (70/20/10 Roboflow) |
| Class | 39 (bệnh + pest + beneficial + healthy) |
| Augment | Ultralytics động (mosaic, fliplr, hsv, scale) |
| Cache | RAM (`cache=True`) |
| data.yaml | `data/citrus/data.yaml` (metadata only, ảnh ở Roboflow/Kaggle) |

**File liên quan:**
- [data/citrus/data.yaml](../../data/citrus/data.yaml) — 39 class names + cấu hình
- [data/citrus/README.dataset.txt](../../data/citrus/README.dataset.txt) — mô tả Roboflow
- [notebooks/train_yolov8n_drone_kaggle.ipynb](../../notebooks/train_yolov8n_drone_kaggle.ipynb) — cell-4 download Roboflow, cell-5 find data.yaml

**Tiếp theo:** [04_export_deploy.md](04_export_deploy.md) — export ONNX, convert kmodel, deploy K230.
