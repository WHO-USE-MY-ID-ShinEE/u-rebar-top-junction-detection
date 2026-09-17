# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
tif = imread_any(os.path.join(data, "station_1_output_depth_raw.tif")).astype(np.float32)
png = imread_any(os.path.join(data, "station_1_output_image_left.png"))
g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
H, W = g.shape
out = r"D:\work\gygj\_probe\preview"; os.makedirs(out, exist_ok=True)

for tag, rot in [("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE), ("cw", cv2.ROTATE_90_CLOCKWISE)]:
    d = cv2.rotate(tif, rot)
    # map depth into gray pixel grid without resizing (both 1440x1080 after rotate)
    base = cv2.normalize(np.clip(g, 0, 255), None, 0, 255, cv2.NORM_MINMAX)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    fin = np.isfinite(d) & (d > 0)
    near = fin & (d < 0.83)     # band A
    far  = fin & (d >= 0.83)    # band B
    vis[near] = (vis[near] * 0.35 + np.array([0, 0, 255]) * 0.65).astype(np.uint8)
    vis[far]  = (vis[far]  * 0.35 + np.array([255, 128, 0]) * 0.65).astype(np.uint8)
    cv2.imwrite(os.path.join(out, f"layer_{tag}.png"), cv2.resize(vis, (W//2, H//2)))
    print(tag, "near(near-band) px:", int(near.sum()), "far-band px:", int(far.sum()))
