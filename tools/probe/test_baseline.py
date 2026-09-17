# -*- coding: utf-8 -*-
"""独立基准：沿列/行统计顶层掩膜的"短游程"个数，作为横筋/竖筋条数的参考。

原理：沿某一列自上而下扫描顶层掩膜，横筋（水平走向）只贡献 ~17px 的短游程，
竖筋贡献很长的游程。取所有列上短游程个数的众数，即横筋条数。行方向同理得竖筋条数。
这个结果完全不依赖骨架/Hough/聚类，可以用来校验检测器。
"""
import sys, numpy as np, cv2
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"


def runs_1d(v):
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    st = np.where(d == 1)[0]
    en = np.where(d == -1)[0]
    return en - st


def baseline(mask, axis, short_lo=8, short_hi=45):
    m = mask if axis == 0 else mask.T
    counts = []
    for i in range(m.shape[1]):
        r = runs_1d(m[:, i])
        counts.append(int(((r >= short_lo) & (r <= short_hi)).sum()))
    counts = np.array(counts)
    vals, freq = np.unique(counts, return_counts=True)
    return int(vals[np.argmax(freq)]), counts


print(f"{'站':>3} | {'基准横':>6} {'检测横':>6} | {'基准竖':>6} {'检测竖':>6} | "
      f"{'基准交叉':>8} {'检测保留':>8}")
for n in range(1, 11):
    gray, d, raw = imgio.load_station(DATA, n)
    split = layers.auto_split_mm(d)
    top, far, _ = layers.split_layers(d, split)
    top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    bh, _ = baseline(top, axis=0)
    bv, _ = baseline(top, axis=1)
    res = detect.detect_rebar_intersections(d, gray)
    s = res["stats"]
    print(f"{n:>3} | {bh:>6} {s['h_bars']:>6} | {bv:>6} {s['v_bars']:>6} | "
          f"{bh*bv:>8} {s['accepted']:>8}")
