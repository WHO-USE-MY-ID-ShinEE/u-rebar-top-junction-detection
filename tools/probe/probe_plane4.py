# -*- coding: utf-8 -*-
"""画顶层横筋/竖筋的深度趋势剖面，判断残余变化是平面、曲面还是噪声。"""
import sys, numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"
K = 25
kh = cv2.getStructuringElement(cv2.MORPH_RECT, (K, 1))
kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, K))


def profile(x, y, z, axis, nb=24):
    v = x if axis == 0 else y
    edges = np.linspace(v.min(), v.max() + 1, nb + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nb - 1)
    out = np.full(nb, np.nan)
    for i in range(nb):
        s = z[idx == i]
        if s.size > 30:
            out[i] = np.median(s)
    return (edges[:-1] + edges[1:]) / 2, out


stations = [1, 4, 10]
fig, axes = plt.subplots(len(stations), 4, figsize=(22, 4.2 * len(stations)))
for row, n in enumerate(stations):
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    near = (m & (d < split)).astype(np.uint8)
    hm = (cv2.morphologyEx(near, cv2.MORPH_OPEN, kh) > 0) & ~(cv2.morphologyEx(near, cv2.MORPH_OPEN, kv) > 0)
    vm = (cv2.morphologyEx(near, cv2.MORPH_OPEN, kv) > 0) & ~(cv2.morphologyEx(near, cv2.MORPH_OPEN, kh) > 0)

    for j, (tag, mask) in enumerate([("H", hm), ("V", vm)]):
        y, x = np.nonzero(mask)
        z = d[mask].astype(np.float64)
        for k, axis in enumerate([0, 1]):
            cx, cy = profile(x.astype(float), y.astype(float), z, axis)
            ax = axes[row, j * 2 + k]
            ax.plot(cx, cy, "o-", ms=4)
            ax.set_title(f"st{n} {tag}筋  depth vs {'x' if axis==0 else 'y'}", fontsize=10)
            ax.grid(alpha=.3)
            if axis == 0:
                ax.set_ylabel("depth mm")
fig.tight_layout()
fig.savefig(r"D:\work\gygj\tools\probe\preview\plane_profile.png", dpi=90)
print("saved tools/probe/preview/plane_profile.png")
