# -*- coding: utf-8 -*-
"""以文本表格输出顶层横筋/竖筋的深度剖面，判断趋势形状。"""
import sys, numpy as np, cv2
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"
K = 25
kh = cv2.getStructuringElement(cv2.MORPH_RECT, (K, 1))
kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, K))


def profile(v, z, nb=20):
    edges = np.linspace(v.min(), v.max() + 1, nb + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nb - 1)
    return np.array([np.median(z[idx == i]) if (idx == i).sum() > 30 else np.nan
                     for i in range(nb)]), (edges[:-1] + edges[1:]) / 2


for n in [1, 4, 10]:
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    near = (m & (d < split)).astype(np.uint8)
    ho = cv2.morphologyEx(near, cv2.MORPH_OPEN, kh) > 0
    vo = cv2.morphologyEx(near, cv2.MORPH_OPEN, kv) > 0
    print(f"===== station {n}  (H=横筋 V=竖筋, 单位 mm) =====")
    for tag, mask in [("H", ho & ~vo), ("V", vo & ~ho)]:
        y, x = np.nonzero(mask)
        z = d[mask].astype(np.float64)
        px, cx = profile(x.astype(float), z, 12)
        py, cy = profile(y.astype(float), z, 12)
        print(f"  {tag}筋 沿 x: " + " ".join(f"{v:6.0f}" for v in px))
        print(f"        x位置: " + " ".join(f"{v:6.0f}" for v in cx))
        print(f"  {tag}筋 沿 y: " + " ".join(f"{v:6.0f}" for v in py))
        print(f"        y位置: " + " ".join(f"{v:6.0f}" for v in cy))
        print(f"        (沿x极差 {np.nanmax(px)-np.nanmin(px):.0f}mm, "
              f"沿y极差 {np.nanmax(py)-np.nanmin(py):.0f}mm)")
