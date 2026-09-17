# -*- coding: utf-8 -*-
import cv2, numpy as np, os

def imread_any(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

data = r"D:\work\gygj\任务书数据+代码"
for n in range(1, 11):
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif"))
    v = tif.astype(np.float64)
    fin = np.isfinite(v) & (v > 0)
    print(f"station {n:2d}: valid={fin.mean():.3f} "
          f"min={np.nanmin(v[fin]):.2f} p1={np.nanpercentile(v[fin],1):.2f} "
          f"med={np.nanpercentile(v[fin],50):.2f} p99={np.nanpercentile(v[fin],99):.2f} "
          f"max={np.nanmax(v[fin]):.2f}")
