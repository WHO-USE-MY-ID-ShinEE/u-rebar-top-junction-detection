# -*- coding: utf-8 -*-
"""用"短游程中心位置直方图 + 寻峰"独立数出横筋/竖筋条数。

沿列扫描：短游程（属于横筋的截面）的中心 y 位置做直方图 -> 峰数 = 横筋条数。
沿行扫描：短游程的中心 x 位置做直方图 -> 峰数 = 竖筋条数。
比"逐列计数取众数"稳健：碎片化的筋条即使断成几段，中心位置仍然聚在同一个峰上。
"""
import sys, numpy as np, cv2
from scipy.signal import find_peaks
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"


def run_centers(mask, axis, lo=4, hi=60):
    m = mask if axis == 0 else mask.T
    centers = []
    for i in range(m.shape[1]):
        col = m[:, i].astype(np.int8)
        d = np.diff(np.concatenate([[0], col, [0]]))
        st = np.where(d == 1)[0]
        en = np.where(d == -1)[0]
        ln = en - st
        ok = (ln >= lo) & (ln <= hi)
        centers.extend(((st[ok] + en[ok]) / 2.0).tolist())
    return np.array(centers)


def count_peaks(centers, bins, min_dist, min_frac=0.05):
    h, _ = np.histogram(centers, bins=bins)
    if h.max() == 0:
        return 0, h
    pk, _ = find_peaks(h, height=h.max() * min_frac, distance=min_dist)
    return int(pk.size), h


print(f"{'站':>3} | {'基准横':>6} {'检测横':>6} | {'基准竖':>6} {'检测竖':>6} | "
      f"{'基准交叉':>8} {'检测保留':>8}")
for n in range(1, 11):
    gray, d, raw = imgio.load_station(DATA, n)
    split = layers.auto_split_mm(d)
    top, far, _ = layers.split_layers(d, split)
    top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    cy = run_centers(top, axis=0)
    cx = run_centers(top, axis=1)
    bh, _ = count_peaks(cy, np.arange(0, top.shape[0] + 1), 60)
    bv, _ = count_peaks(cx, np.arange(0, top.shape[1] + 1), 60)

    res = detect.detect_rebar_intersections(d, gray)
    s = res["stats"]
    print(f"{n:>3} | {bh:>6} {s['h_bars']:>6} | {bv:>6} {s['v_bars']:>6} | "
          f"{bh*bv:>8} {s['accepted']:>8}")
