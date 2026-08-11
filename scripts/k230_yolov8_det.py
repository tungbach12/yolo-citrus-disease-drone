# -*- coding: utf-8 -*-
# YOLOv8 detect trên CanMV K230
# Tham khảo: CanMV K230 - Bài 9 - Model Deployment (Cat_and_dog_detection.py)
#
# CÁCH DÙNG:
#   1) Copy best.kmodel + file này lên thẻ SD của K230 (thư mục /sdcard)
#   2) Mở file này trong CanMV IDE K230
#   3) Bấm nút Connect (icon 1) rồi Run (icon 2)
#
# BẮT BUỘC SỬA 3 CHỖ: kmodel_path, labels, model_input_size

from libs.PipeLine import PipeLine, ScopedTiming
from libs.YOLO import YOLOv8
import os, sys, gc

if __name__ == "__main__":
    # (1) ĐỔI THÀNH TÊN FILE kmodel CỦA BẠN (/sdcard/<tên>)
    kmodel_path = "/sdcard/best.kmodel"

    # (2) ĐỔI THÀNH DANH SÁCH CLASS, ĐÚNG THỨ TỰ NHƯ data.yaml
    labels = [
        "beneficial_insect", "black_aphid", "brown_banded_tortrix", "citrus_anthracnose",
        "citrus_aphid", "citrus_black_spot", "citrus_brown_spot", "citrus_canker",
        "citrus_exocortis", "citrus_fruit_fly", "citrus_greasy_spot", "citrus_huanglongbing",
        "citrus_leafminer", "citrus_leprosis", "citrus_longhorned_beetle", "citrus_melanose",
        "citrus_powdery_mildew", "citrus_psyllid", "citrus_red_mite", "citrus_rot",
        "citrus_rust_mite", "citrus_scab", "citrus_sooty_mold", "citrus_swallowtail",
        "citrus_thrips", "citrus_whitefly", "healthy_leaf", "honeybee",
        "lacewing", "ladybug", "other_disease", "other_pest",
        "other_scale_insect", "other_slug_moth", "red_wax_scale", "spider",
        "spring_weevil", "thripidae", "wasp",
    ]

    # (3) ĐỔI THÀNH imgsz LÚC EXPORT ONNX (đúng IMGSZ trong notebook, mặc định 640)
    model_input_size = [640, 640]

    # Ngưỡng tin cậy / NMS (tùy chỉnh)
    confidence_threshold = 0.3
    nms_threshold = 0.7

    # Hoạt động camera ở 1280x720, hiển thị lên màn 800x480
    rgb888p_size = [1280, 720]
    display_size = [800, 480]

    # Khởi tạo đường truyền ảnh camera -> màn hình
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode="lcd")
    pl.create()

    # Đối tượng YOLOv8 nhận diện
    yolo = YOLOv8(
        task_type="detect",
        mode="video",
        kmodel_path=kmodel_path,
        labels=labels,
        rgb888p_size=rgb888p_size,
        model_input_size=model_input_size,
        display_size=display_size,
        conf_thresh=confidence_threshold,
        nms_thresh=nms_threshold,
        debug_mode=0,
    )
    yolo.config_preprocess()

    try:
        print("YOLOv8 model loaded successfully, starting detection...")
        while True:
            os.exitpoint()

            with ScopedTiming("total", 1):
                img = pl.get_frame()       # lấy 1 khung hình camera
                res = yolo.run(img)        # chạy inference
                yolo.draw_result(res, pl.osd_img)  # vẽ khung kết quả
                pl.show_image()            # hiển thị
                gc.collect()               # thu hồi bộ nhớ
    except Exception as e:
        sys.print_exception(e)
    finally:
        yolo.deinit()
        pl.destroy()
        print("Program stopped")