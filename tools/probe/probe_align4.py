# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
out = r"D:\work\gygj\_probe\preview"

def gray_feature(g):
    c = cv2.createCLAHE(3.0, (8, 8)).apply(g)
    # texture energy: local std via box filters
    f = c.astype(np.float32)
    m = cv2.boxFilter(f, -1, (9, 9))
    v = cv2.boxFilter(f*f, -1, (9, 9)) - m*m
    v = np.clip(v, 0, None)
    return c, v

for n in [1, 4, 7, 10]:
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif")).astype(np.float32)
    png = imread_any(os.path.join(data, f"station_{n}_output_image_left.png"))
    g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    d = cv2.rotate(tif, cv2.ROTATE_90_COUNTERCLOCKWISE)
    c, tex = gray_feature(g)
    fin = (np.isfinite(d) & (d > 0)).astype(np.float32)
    fin_f = cv2.boxFilter(fin, -1, (9, 9))
    # NCC between "is rebar surface" mask and texture energy
    a = fin_f - fin_f.mean(); b = tex - tex.mean()
    best, bb = -9, None
    for dy in range(-20, 21):
        for dx in range(-20, 21):
            bs = np.roll(np.roll(b, dy, 0), dx, 1)
            s = float((a*bs).sum())
            if s > best: best, bb = s, (dy, dx)
    print(f"station {n}: texture-NCC peak at (dy,dx)={bb}   score={best:.4e}   "
          f"@(0,0)={float((a*b).sum()):.4e}")

# visual: boundary overlay at (0,0) vs best shift, station 1
tif = imread_any(os.path.join(data, "station_1_output_depth_raw.tif")).astype(np.float32)
png = imread_any(os.path.join(data, "station_1_output_image_left.png"))
g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
d = cv2.rotate(tif, cv2.ROTATE_90_COUNTERCLOCKWISE)
c = cv2.createCLAHE(3.0, (8, 8)).apply(g)
fin = (np.isfinite(d) & (d > 0)).astype(np.uint8)*255
bd = cv2.Canny(fin, 50, 150) > 0
y0,y1,x0,x1 = 380, 700, 260, 600
tiles = []
for tag, (dy, dx) in [("shift(0,0)", (0,0)), ("shift(-9,0)", (-9,0)), ("shift(-9,-11)", (-9,-11))]:
    sh = np.roll(np.roll(bd, dy, 0), dx, 1)
    vis = cv2.cvtColor(c[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    m = sh[y0:y1, x0:x1]
    vis[m] = (0, 0, 255)
    cv2.putText(vis, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    tiles.append(vis)
    tiles.append(np.full((y1-y0, 6, 3), 255, np.uint8))
z = np.hstack(tiles[:-1])
z = cv2.resize(z, (z.shape[1]*3//2, z.shape[0]*3//2), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(out, "align_check.png"), z)
print("saved", z.shape)
