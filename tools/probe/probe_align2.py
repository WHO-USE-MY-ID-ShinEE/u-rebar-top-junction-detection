# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
res = {}
for n in range(1, 11):
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif")).astype(np.float32)
    png = imread_any(os.path.join(data, f"station_{n}_output_image_left.png"))
    g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    ge = (cv2.Canny(g, 30, 100) > 0).astype(np.float32)
    for tag, rot in [("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE), ("cw", cv2.ROTATE_90_CLOCKWISE)]:
        d = cv2.rotate(tif, rot)
        fin = (np.isfinite(d) & (d > 0)).astype(np.uint8) * 255
        de = cv2.Canny(fin, 50, 150) > 0
        de = cv2.dilate(de.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        best = 0.0
        for dy in range(-25, 26, 5):
            for dx in range(-25, 26, 5):
                sh = np.roll(np.roll(de, dy, 0), dx, 1)
                s = float((ge * sh).sum()) / max(ge.sum(), 1)
                best = max(best, s)
        res.setdefault(tag, []).append(best)
for tag, v in res.items():
    print(f"{tag}:  mean={np.mean(v):.4f}  per-station={[round(x,3) for x in v]}")
