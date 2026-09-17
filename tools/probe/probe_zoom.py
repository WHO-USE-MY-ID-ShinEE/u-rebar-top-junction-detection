# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
out = r"D:\work\gygj\_probe\preview"
tif = imread_any(os.path.join(data, "station_1_output_depth_raw.tif")).astype(np.float32)
png = imread_any(os.path.join(data, "station_1_output_image_left.png"))
g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
d = cv2.rotate(tif, cv2.ROTATE_90_COUNTERCLOCKWISE)
g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

# row profile of depth through a vertical bar region, and column profile
print("column x=300, y=100..260 depth (mm):")
print(np.round(d[100:260:4, 300]*1000, 1))
print()
print("row y=500, x=200..360 depth (mm):")
print(np.round(d[500, 200:360:4]*1000, 1))

# zoom crop: gray | depth colormap, aligned
y0,y1,x0,x1 = 380, 720, 260, 620
crop_g = cv2.cvtColor(g8[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
dc = d[y0:y1, x0:x1]
fin = np.isfinite(dc) & (dc>0)
dn = np.zeros_like(dc, np.uint8)
dn[fin] = np.clip((dc[fin]-0.70)/(0.96-0.70)*255, 0, 255).astype(np.uint8)
crop_d = cv2.applyColorMap(cv2.resize(dn,(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
zoom = np.hstack([crop_g, np.full((y1-y0,8,3),255,np.uint8), crop_d])
zoom = cv2.resize(zoom, (zoom.shape[1]*2, zoom.shape[0]*2), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(out,"zoom1.png"), zoom)
print("zoom saved", zoom.shape)
