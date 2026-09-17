# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
for n in [1, 4, 7, 10]:
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif")).astype(np.float32)
    png = imread_any(os.path.join(data, f"station_{n}_output_image_left.png"))
    g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    d = cv2.rotate(tif, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ge = cv2.Canny(g, 30, 100) > 0
    fin = (np.isfinite(d) & (d > 0)).astype(np.uint8) * 255
    de = cv2.Canny(fin, 50, 150) > 0
    de = cv2.dilate(de.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    best, bb = -1, None
    for dy in range(-20, 21):
        for dx in range(-20, 21):
            sh = np.roll(np.roll(de, dy, 0), dx, 1)
            s = float((ge & sh).sum()) / ge.sum()
            if s > best: best, bb = s, (dy, dx)
    # 3x3 sub-grid around the peak
    top = []
    for dy in range(bb[0]-1, bb[0]+2):
        for dx in range(bb[1]-1, bb[1]+2):
            sh = np.roll(np.roll(de, dy, 0), dx, 1)
            top.append((round(float((ge & sh).sum())/ge.sum(), 4), dy, dx))
    print(f"station {n}: peak score={best:.4f} at (dy,dx)={bb}  baseline@(0,0)="
          f"{float((ge & de).sum())/ge.sum():.4f}")
