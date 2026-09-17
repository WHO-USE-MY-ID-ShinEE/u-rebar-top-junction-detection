# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
tif = imread_any(os.path.join(data, "station_1_output_depth_raw.tif")).astype(np.float32)
png = imread_any(os.path.join(data, "station_1_output_image_left.png"))
g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
print("depth", tif.shape, "gray", g.shape)
d_cw = cv2.rotate(tif, cv2.ROTATE_90_CLOCKWISE)
d_ccw = cv2.rotate(tif, cv2.ROTATE_90_COUNTERCLOCKWISE)
print("cw ", d_cw.shape, "ccw", d_ccw.shape)

# depth histogram: is it quantised into discrete layers?
v = d_cw[np.isfinite(d_cw) & (d_cw > 0)]
h, edges = np.histogram(v, bins=120, range=(0.6, 1.1))
for c, e in zip(h, edges):
    if c > v.size * 0.003:
        print(f"  {e:.3f} m  {c:8d}  {'#'*int(c/v.size*400)}")

# edge-overlap alignment score for both rotations
def score(depth):
    fin = (np.isfinite(depth) & (depth > 0)).astype(np.uint8)
    fin = cv2.resize(fin*255, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    gx = cv2.Canny(g, 40, 120) > 0
    dx = cv2.Canny(fin, 50, 150) > 0
    dx = cv2.dilate(dx.astype(np.uint8), np.ones((7,7), np.uint8)) > 0
    return float((gx & dx).sum()) / max(int(gx.sum()), 1)
print("edge overlap CW :", round(score(d_cw), 4))
print("edge overlap CCW:", round(score(d_ccw), 4))
print("edge overlap raw(T):", round(score(cv2.rotate(tif, cv2.ROTATE_180)), 4))
