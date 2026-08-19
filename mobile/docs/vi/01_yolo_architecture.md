# 01 — Kiến trúc YOLO & YOLOv8

> Tài liệu tự học về kiến trúc mô hình YOLO, tập trung vào YOLOv8 (biến thể `n` mà project này dùng để deploy lên drone K230). Đọc trước khi sang [02_training_process.md](02_training_process.md).

## 1. YOLO là gì — "You Only Look Once"

**YOLO** (You Only Look Once) là họ mô hình **object detection** (nhận diện vật thể) **one-stage** (một giai đoạn). Ý tưởng cốt lõi: nhìn ảnh **một lần** duy nhất qua mạng neural → ra luôn cả danh sách vật thể + vị trí + lớp.

### One-stage vs Two-stage

| | Two-stage (R-CNN, Faster R-CNN) | One-stage (YOLO, SSD) |
|---|---|---|
| Cách làm | Bước 1: tìm ~2000 vùng có thể có vật thể (region proposals). Bước 2: classify từng vùng. | Chạy thẳng ảnh qua mạng → ra toàn bộ box + class cùng lúc. |
| Độ chính xác | Cao hơn một chút (đặc biệt vật thể nhỏ) | Thấp hơn một chút |
| Tốc độ | Chậm (cả chục FPS) | Nhanh (vài trăm FPS) |
| Phù hợp | Y tế, ảnh vệ tinh — cần chính xác, không cần thời gian thực | Drone, camera, self-driving — cần thời gian thực |

**Vì sao drone dùng YOLO?** Drone bay liên tục, mỗi giây có ~30 khung hình. Nếu mô hình chậm hơn tốc độ frame → độ trễ tích tụ → drone phản ứng muộn → mất vật thể. YOLO đủ nhanh để chạy **realtime** trên phần cứng yếu (K230, Raspberry Pi).

> **Trade-off thực tế:** YOLOv8n trên K230 chạy ~10-20 FPS — đủ cho drone bay chậm quét ruộng cam. Nếu cần chính xác tuyệt đối (bệnh ở giai đoạn rất sớm, vết nhỏ 5px) thì two-stage chính xác hơn, nhưng không realtime được trên K230.

---

## 2. Kiến trúc YOLOv8 — 3 phần: Backbone → Neck → Head

Mọi mô hình detection hiện đại có cấu trúc 3 khối:

```
Ảnh đầu vào (640×640)
      ↓
  [Backbone]  — trích đặc trưng (features) từ ảnh
      ↓
  [Neck]      — gộp features ở nhiều尺度 (multi-scale)
      ↓
  [Head]      — ra dự đoán: box + class cho từng vật thể
```

### 2.1 Backbone — CSPDarknet

Backbone là "trạm quan sát": nhận ảnh thô, trích xuất **features** (đặc trưng) ở nhiều mức độ trừu tượng:
- **Layer sớm**: phát hiện cạnh, góc, texture (chi tiết nhỏ).
- **Layer muộn**: phát hiện cấu trúc ngữ nghĩa cao (vật thể phức tạp: con côn trùng, vệt bệnh).

YOLOv8 dùng **CSPDarknet** (Cross-Stage Partial Darknet). "CSP" = chia feature map làm 2 nhánh, xử lý song song rồi gộp lại → **giảm tính toán lặp** mà vẫn giữ thông tin. Đó là lý do YOLOv8n nhẹ hơn mà vẫn mạnh.

### 2.2 Neck — PANet (FPN + PAN)

Neck làm việc **multi-scale detection** — phát hiện vật thể to lẫn nhỏ trên cùng ảnh:
- **FPN** (Feature Pyramid Network): truyền thông tin từ layer muộn (bức tranh lớn, ngữ cảnh) xuống layer sớm (chi tiết) → giúp phát hiện vật thể to.
- **PAN** (Path Aggregation Network): truyền ngược từ layer sớm lên muộn → giúp giữ chi tiết tinh cho vật thể nhỏ.

Trên drone chụp từ trên cao, vật thể (lá cam bệnh) thường **rất nhỏ** so với ảnh 640×640 → PANet quan trọng. Đó là vì sao YOLOv8 phát hiện nhỏ tốt hơn nhiều bản YOLO cũ.

### 2.3 Head — Decoupled, Anchor-free

**Head** là "bộ phán quyết": nhận features đã gộp, ra dự đoán cuối.

**Decoupled head** (tách rời): YOLOv8 tách luồng dự đoán **box** (vị trí) và **class** (loại) thành 2 nhánh riêng. Bản cũ (YOLOv5) gộp chung → gradient xung đột, khó hội tụ. Tách ra → mỗi nhánh học tốt nhiệm vụ riêng.

**Anchor-free** (không dùng anchor box): đây là thay đổi lớn nhất của YOLOv8. Xem bên dưới.

---

## 3. "Anchor-free" là gì — và vì sao YOLOv8 bỏ anchor

### Anchor box là gì (bản cũ)

YOLOv5 và trước đó dùng **anchor box** — các "khung mẫu" kích thước định sẵn (vd: `[10,13], [16,30], [33,23]` cho ảnh nhỏ). Mô hình học **điều chỉnh** anchor này cho vừa vật thể, thay vì dự đoán box từ đầu.

**Vấn đề:**
- Anchor phải **chọn trước** dựa trên dataset (cluster kích thước GT) → đổi dataset phải tính lại anchor.
- Với dataset nhỏ/khó, anchor sai → mô hình kém.
- Thêm bước xử lý phức tạp.

### YOLOv8: anchor-free

YOLOv8 dự đoán **trực tiếp** 4 giá trị: khoảng cách từ điểm trung tâm cell đến 4 cạnh (trên/dưới/trái/phải) của box. Không cần anchor.

**Lợi ích:**
- Không phải tính anchor, đổi dataset không phải chỉnh gì.
- Phát hiện vật thể kích thước bất kỳ (không bị giới hạn bởi anchor mẫu).
- Đơn giản hơn, nhanh hơn.

**Đây là vì sao bạn không thấy `anchors` trong `data.yaml` của project — YOLOv8 không cần.**

---

## 4. Kích thước mô hình — n / s / m / l / x

YOOLOv8 có 5 kích thước, cùng kiến trúc nhưng depth (số layer) + width (số channel) khác nhau:

| Biến thể | Tham số | GFLOPs | Mục đích |
|---|---|---|---|
| **YOLOv8n** (nano) | 3.0M | 8.1 | Edge device (K230, Pi, mobile) |
| YOLOv8s (small) | 11.2M | 28.6 | Server nhẹ, Jetson |
| YOLOv8m (medium) | 25.9M | 78.9 | Workstation GPU |
| YOLOv8l (large) | 43.7M | 165.2 | GPU mạnh |
| YOLOv8x (xlarge) | 68.2M | 257.8 | Server, competition |

### Vì sao project này dùng `n`?

K230 có giới hạn bộ nhớ KPU (~1GB region) và sức tính toán hạn chế. Mô hình lớn hơn → kmodel lớn hơn → không nạp hoặc chạy quá chậm.

Project này: `best.pt` = 5.4 MB, `best.kmodel` = 11.6 MB. Trên K230 chạy được ở ~10-20 FPS. Nếu dùng `s` → kmodel ~50MB, có thể không nạp hoặc chỉ 2-3 FPS.

> **Trade-off:** `n` ít tham số hơn → chính xác thấp hơn `s/m`. Nhưng trên K230 không có lựa chọn — `n` là điểm sweet spot giữa "chạy được" và "chạy nhanh".

**Kiểm chứng thực tế:** khi load `models/best.pt` bằng ultralytics, output summary:
```
Model summary (fused): 73 layers, 3,013,253 parameters, 0 gradients, 8.1 GFLOPs
```
→ đúng đặc tả YOLOv8n.

---

## 5. Input shape & Output shape — đọc số liệu

### Input: `(1, 3, 640, 640)` — BCHW

| Chỉ số | Ý nghĩa |
|---|---|
| `1` | Batch size — 1 ảnh mỗi lần infer (K230 chạy từng frame) |
| `3` | 3 kênh màu RGB |
| `640` | Chiều cao (pixel) |
| `640` | Chiều rộng (pixel) |

**BCHW** = Batch/Channel/Height/Width — thứ tự trục mà PyTorch/ONNX dùng. (OpenCV cho ảnh HWC, phải transpose trước khi đưa vào.)

`640` là `IMGSZ` trong notebook — **phải khớp** giữa train, export ONNX, và `model_input_size` trong `k230_yolov8_det.py`. Đổi một chỗ mà không đổi chỗ khác → mô hình sai.

### Output: `(1, 43, 8400)`

Đây là điểm hay. Đọc từ phải qua trái theo cách mô hình "nghĩ":

**`8400`** = tổng số cell trên 3 scale (grid cells). YOLOv8 chia ảnh thành lưới ở 3 độ phân giải:
- 80×80 (stride 8) — phát hiện vật thể **nhỏ** (lá bệnh từ xa)
- 40×40 (stride 16) — vật thể **vừa**
- 20×20 (stride 32) — vật thể **to**

```
80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 = 8400 cells
```
Mỗi cell dự đoán **1 vật thể** (anchor-free) → tối đa 8400 box trên 1 ảnh.

**`43`** = số giá trị mỗi cell dự đoán:
- `4`: tọa độ box — khoảng cách đến 4 cạnh (top/bottom/left/right) từ tâm cell
- `39`: score cho 39 class citrus (xác suất từng loại bệnh/dịch hại)

```
4 + 39 = 43
```

**`1`** = batch size.

> **Hệ quả:** Nếu bạn đổi dataset sang N class khác, output shape tự đổi thành `(1, 4+N, 8400)`. Không phải sửa code. Đây là vì sao YOLOv8 "linh hoạt" — chỉ cần `data.yaml` đúng `nc`.

---

## 6. Post-processing — từ raw output ra box cuối

Output `(1, 43, 8400)` là raw — cần 3 bước mới ra box dùng được:

1. **Decode box**: raw 4 giá trị (khoảng cách tâm cell → cạnh) → tọa độ `xyxy` (x1,y1,x2,y2) pixel thực.
2. **Lọc confidence**: cell nào có max(class_scores) < `conf_thresh` (vd 0.3) → bỏ. Còn lại vài chục box.
3. **NMS** (Non-Maximum Suppression): nhiều cell cùng phát hiện **1 vật thể** → box chồng nhau. NMS giữ box confidence cao nhất, bỏ box dư (IoU > `nms_thresh`, vd 0.7).

Trong `k230_yolov8_det.py`:
```python
confidence_threshold = 0.3
nms_threshold = 0.7
```

**Vì sao NMS cần thiết?** Vì anchor-free + 3 scale, một con rệp sáp đỏ có thể bị cell 80×80, cell 40×40, và cell 20×20 cùng phát hiện → 3 box chồng. NMS gộp thành 1.

---

## 7. Tổng kết — sơ đồ luồng dữ liệu

```
Camera K230 (1280×720)
   ↓ resize + letterbox
Ảnh 640×640 RGB
   ↓ [Backbone CSPDarknet]
Feature maps multi-scale
   ↓ [Neck PANet]
Feature gộp (80/40/20 cell)
   ↓ [Head anchor-free]
Raw output (1, 43, 8400)
   ↓ decode + conf filter + NMS
Box cuối: [x1,y1,x2,y2, class_id, confidence] × N
   ↓ vẽ lên OSD
Hiển thị màn 800×480
```

**File liên quan trong project:**
- `models/best.pt` — checkpoint PyTorch (chạy được trên laptop bằng ultralytics)
- `models/best.onnx` — bản trung gian, cùng kiến trúc, chạy được bằng onnxruntime
- `models/best.kmodel` — bản biên dịch cho KPU K230, cùng kiến trúc nhưng đã lượng tử + tối ưu phần cứng

---

## 8. Tham khảo

- Paper YOLOv8: trang Ultralytics `docs.ultralytics.com/models/yolov8/`
- So sánh anchor vs anchor-free: `docs.ultralytics.com/anchor-free-detection/`
- Netron (visualize kiến trúc ONNX): `netron.app` — mở `models/best.onnx` để xem đồ thị layer-by-layer

**Tiếp theo:** [02_training_process.md](02_training_process.md) — quá trình train, loss, optimizer, augmentation, metrics.
