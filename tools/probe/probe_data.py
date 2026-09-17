# -*- coding: utf-8 -*-
import cv2, numpy as np, os

def imread_any(path, flags=cv2.IMREAD_UNCHANGED):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, flags)

base = r"D:\work\gygj"
data = os.path.join(base, "任务书数据+代码")
for n in [1, 5, 10]:
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif"))
    png = imread_any(os.path.join(data, f"station_{n}_output_image_left.png"))
    print("station", n)
    print("  tif:", tif.shape, tif.dtype, tif.min(), tif.max())
    print("  png:", png.shape, png.dtype, png.min(), png.max())
    v = tif.astype(np.float64)
    print("  pct:", np.percentile(v, [0,1,5,25,50,75,95,99,100]).round(3))
    print("  nonzero frac:", round(float((v>0).mean()),4))
