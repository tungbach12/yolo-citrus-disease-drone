# -*- coding: utf-8 -*-
# Kaggle web UI downloaded best.zip / last.zip but they were the *contents* of
# the .pt PyTorch archive (unpacked), not the model file itself.
# Re-zip the folder contents flat into proper .pt files.
import os, zipfile

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_out", "models")

for folder, outname in [("best", "best.pt"), ("last", "last.pt")]:
    src = os.path.join(BASE, folder)
    dst = os.path.join(BASE, outname)
    if not os.path.isdir(src):
        print(f"[SKIP] {src} không tồn tại")
        continue
    # PyTorch .pt archives store entries under a top-level folder named after
    # the archive (e.g. "best/"). A flat zip is rejected by PyTorchFileReader.
    prefix = folder + "/"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src):
            for fn in files:
                fp = os.path.join(root, fn)
                arc = prefix + os.path.relpath(fp, src).replace(os.sep, "/")
                zf.write(fp, arc)
    print(f"[OK] {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")
