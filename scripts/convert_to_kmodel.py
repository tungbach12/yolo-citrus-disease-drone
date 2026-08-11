#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Convert ONNX -> KModel (.kmodel) cho CanMV K230
# nncase 2.10.0 API (Compiler nhận compile_options trong constructor,
# import_onnx nhận bytes + ImportOptions, không còn input_layout riêng).
#
# Cài đặt (Windows):
#   pip install nncase==2.10.0
#   curl -L -O https://github.com/kendryte/nncase/releases/download/v2.10.0/nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
#   pip install nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
#   (cần .NET 7.0 x64 runtime)

import os
import onnx
import nncase

ONNX_PATH = "../models/best.onnx"
KMODEL_PATH = "../models/best.kmodel"
INPUT_IMGSZ = 640

# (1) batch size của onnx về 1
print("Bước 1: Chuẩn bị ONNX (batch=1)...")
onnx_model = onnx.load(ONNX_PATH)
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
for o in onnx_model.graph.output:
    o.type.tensor_type.shape.dim[0].dim_value = 1
onnx.save(onnx_model, "best_bs1.onnx")
print("  -> đã tạo best_bs1.onnx")

# (2) biên dịch
print("Bước 2: Biên dịch bằng nncase 2.10.0...")
compile_options = nncase.CompileOptions()
compile_options.target = "k230"
compile_options.input_type = "uint8"
compile_options.input_layout = "NCHW"
compile_options.output_type = "float32"
compile_options.output_layout = "NCHW"
compile_options.preprocess = False
compile_options.input_shape = [1, 3, INPUT_IMGSZ, INPUT_IMGSZ]

compiler = nncase.Compiler(compile_options)
with open("best_bs1.onnx", "rb") as f:
    model_bytes = f.read()
import_options = nncase.ImportOptions()
compiler.import_onnx(model_bytes, import_options)
compiler.compile()

# (3) ghi kmodel
print("Bước 3: Ghi file kmodel...")
kmodel = compiler.gencode_tobytes()
with open(KMODEL_PATH, "wb") as f:
    f.write(kmodel)
print(f"  -> đã tạo {KMODEL_PATH} ({len(kmodel)/1024:.1f} KB)")
print("Done. Copy", KMODEL_PATH, "vào /sdcard của K230.")
