# -*- coding: utf-8 -*-
# Cắt ảnh lớn thành tile 640x640 (có overlap), lưu vào folder, chạy YOLO v2 từng tile.
#
# Khác với drone_detect_sahi.py (SAHI gộp NMS tự động, không lưu tile),
# script này LƯU từng tile ra folder để bạn xem model "thấy" gì trong mỗi ô.
#
# CÁCH DêNG:
#   python tile_and_detect.py <ảnh hoặc thư mục>
#   CONF=0.15 SLICE=640 OVERLAP=0.25 python tile_and_detect.py image.jpg
#   python tile_and_detect.py "C:\Users\trant\Downloads\photo.jpg"
#
# Output: tạo folder <tên ảnh>_tiles/ bên cạnh ảnh, chứa:
#   raw/       — tile gốc (chưa detect)
#   detected/  — tile đã vẽ box + nhãn
#   summary.txt — thống kê detection từng tile

import os
import sys

# ====== CẤU HÌNH ======
MODEL_PATH = os.environ.get("MODEL_PATH", "../models/v2/best.pt")
CONF = float(os.environ.get("CONF", "0.15"))
SLICE = int(os.environ.get("SLICE", "640"))
OVERLAP = float(os.environ.get("OVERLAP", "0.25"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
ENHANCE = os.environ.get("ENHANCE", "0") in ("1", "true", "True")
IMGSZ = int(os.environ.get("IMGSZ", os.environ.get("SLICE", "640")))
# ======================


def enhance_tile(tile_bgr):
    """Tiền xử lý tăng độ nét (Sharpen) + tăng tương phản (CLAHE) cho tile bị mờ do drone chụp xa."""
    import cv2
    import numpy as np

    # 1. Sharpening (Làm nét biên lá)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(tile_bgr, -1, kernel)

    # 2. CLAHE trên kênh L (LAB color space)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


def slice_image(img_bgr, slice_size=SLICE, overlap=OVERLAP):
    """Cắt ảnh BGR thành list các (x_offset, y_offset, tile_bgr).
    Đảm bảo phủ kín mép phải + mép dưới (thêm tile sát biên)."""
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    step = max(int(slice_size * (1 - overlap)), 1)

    def _positions(length):
        if length <= slice_size:
            return [0]
        pos = list(range(0, length - slice_size + 1, step))
        if pos[-1] != length - slice_size:
            pos.append(length - slice_size)
        return pos

    xs = _positions(w)
    ys = _positions(h)
    tiles = []
    for y in ys:
        for x in xs:
            tile = img_bgr[y:y + slice_size, x:x + slice_size]
            # pad lên slice_size nếu tile mép nhỏ hơn
            if tile.shape[0] < slice_size or tile.shape[1] < slice_size:
                pad_b = slice_size - tile.shape[0]
                pad_r = slice_size - tile.shape[1]
                tile = cv2.copyMakeBorder(tile, 0, pad_b, 0, pad_r,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            tiles.append((x, y, tile))
    return tiles, (w, h)


def run_detection(model, tile_bgr):
    """Chạy YOLO trên 1 tile BGR. Trả về (annotated_img, list_of(label,score,x1,y1,x2,y2))."""
    import cv2
    input_tile = enhance_tile(tile_bgr) if ENHANCE else tile_bgr
    results = model.predict(input_tile, conf=CONF, device=DEVICE, verbose=False, imgsz=IMGSZ)
    annotated = tile_bgr.copy()
    dets = []
    r = results[0]
    if r.boxes is None:
        return annotated, dets
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        dets.append((label, score, x1, y1, x2, y2))
    return annotated, dets


def process_image(model, path, out_root=None):
    import cv2
    img = cv2.imread(path)
    if img is None:
        print(f"  Không đọc được: {path}")
        return None
    h, w = img.shape[:2]
    print(f"Ảnh: {path} ({w}x{h})")

    if out_root is None:
        base = os.path.splitext(os.path.basename(path))[0]
        out_root = os.path.join(os.path.dirname(path), base + "_tiles")

    raw_dir = os.path.join(out_root, "raw")
    det_dir = os.path.join(out_root, "detected")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)

    tiles, (w, h) = slice_image(img)
    print(f"  Cắt thành {len(tiles)} tile ({SLICE}x{SLICE}, overlap={OVERLAP})")
    print(f"  Lưu vào: {out_root}")

    summary_lines = [f"Ảnh: {os.path.basename(path)} ({w}x{h})",
                     f"Tile: {SLICE}x{SLICE} | overlap={OVERLAP} | conf={CONF}",
                     f"Tổng số tile: {len(tiles)}", ""]

    total_dets = 0
    from collections import Counter
    class_counts = Counter()

    for i, (x, y, tile) in enumerate(tiles):
        annotated, dets = run_detection(model, tile)
        fname = f"tile_{i:04d}_x{x:05d}_y{y:05d}.jpg"
        cv2.imwrite(os.path.join(raw_dir, fname), tile)
        cv2.imwrite(os.path.join(det_dir, fname), annotated)
        if dets:
            total_dets += len(dets)
            for label, score, *_ in dets:
                class_counts[label] += 1
            det_str = "; ".join(f"{l} {s:.2f}" for l, s, *_ in dets)
            summary_lines.append(f"{fname} [offset x={x},y={y}]: {det_str}")
            print(f"  {fname}: {len(dets)} det")
        else:
            summary_lines.append(f"{fname} [offset x={x},y={y}]: (không có)")

    summary_lines.append("")
    summary_lines.append(f"TỔNG: {total_dets} detections trên {len(tiles)} tile")
    summary_lines.append("Phân bố class:")
    for label, cnt in class_counts.most_common():
        summary_lines.append(f"  {label}: {cnt}")

    with open(os.path.join(out_root, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"  -> {total_dets} detections | {len(class_counts)} class")
    print(f"  summary.txt đã lưu")
    return out_root


def main():
    from ultralytics import YOLO

    if len(sys.argv) < 2:
        print("Cách dùng: python tile_and_detect.py <ảnh hoặc thư mục>")
        sys.exit(1)

    source = sys.argv[1]
    print(f"Model: {MODEL_PATH} | conf={CONF} | slice={SLICE} | overlap={OVERLAP} | enhance={ENHANCE} | imgsz={IMGSZ} | device={DEVICE}")
    model = YOLO(MODEL_PATH)
    print(f"Loaded. Classes: {len(model.names)}")

    import glob
    if os.path.isdir(source):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        imgs = []
        for e in exts:
            imgs.extend(glob.glob(os.path.join(source, e)))
        if not imgs:
            print(f"Không có ảnh trong: {source}")
            return
        print(f"Tìm thấy {len(imgs)} ảnh")
        for p in sorted(imgs):
            process_image(model, p)
    elif os.path.isfile(source):
        process_image(model, source)
    else:
        print(f"Không tìm thấy: {source}")


if __name__ == "__main__":
    main()
