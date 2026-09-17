# -*- coding: utf-8 -*-
import cv2, numpy as np, os

def imread_any(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

data = r"D:\work\gygj\任务书数据+代码"
out = r"D:\work\gygj\_probe\preview"
os.makedirs(out, exist_ok=True)
for n in range(1, 11):
    png = imread_any(os.path.join(data, f"station_{n}_output_image_left.png"))
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif")).astype(np.float32)
    g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (540, 720), interpolation=cv2.INTER_AREA)
    gray3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # depth rotated 90 CCW to match gray orientation (per v8 code)
    d = cv2.rotate(tif, cv2.ROTATE_90_CLOCKWISE)
    fin = np.isfinite(d) & (d > 0)
    dn = np.zeros_like(d, np.uint8)
    if fin.any():
        lo, hi = np.percentile(d[fin], 2), np.percentile(d[fin], 98)
        dn[fin] = np.clip((d[fin] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    dn = cv2.resize(dn, (540, 720), interpolation=cv2.INTER_NEAREST)
    col = cv2.applyColorMap(dn, cv2.COLORMAP_JET)
    pair = np.hstack([gray3, np.full((720, 10, 3), 255, np.uint8), col])
    cv2.imwrite(os.path.join(out, f"pair_{n}.png"), cv2.resize(pair, (1080, 720)))
print("ok")
