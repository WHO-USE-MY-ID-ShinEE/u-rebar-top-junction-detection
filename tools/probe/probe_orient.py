# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
K = 15
kh = cv2.getStructuringElement(cv2.MORPH_RECT, (K, 1))
kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, K))
for n in [1, 5, 10]:
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif")).astype(np.float32)
    d = cv2.rotate(tif, cv2.ROTATE_90_COUNTERCLOCKWISE)
    fin = (np.isfinite(d) & (d > 0)).astype(np.uint8)
    print(f"--- station {n} ---  total valid px = {int(fin.sum())}")
    bands = [("A <810mm", fin & (d*1000 < 810)),
             ("B 810-870", fin & (d*1000 >= 810) & (d*1000 < 870)),
             ("C >=870mm", fin & (d*1000 >= 870))]
    for name, m in bands:
        m = m.astype(np.uint8)
        if m.sum() < 100:
            print(f"  {name}: empty"); continue
        hh = cv2.morphologyEx(m, cv2.MORPH_OPEN, kh).sum()
        vv = cv2.morphologyEx(m, cv2.MORPH_OPEN, kv).sum()
        dep = d[m > 0]
        print(f"  {name}: px={int(m.sum()):7d} ({m.sum()/fin.sum()*100:4.1f}%) "
              f"median={np.median(dep)*1000:6.1f}mm  Hopen={int(hh):6d} Vopen={int(vv):6d} "
              f"H/V={hh/max(vv,1):5.2f}")
