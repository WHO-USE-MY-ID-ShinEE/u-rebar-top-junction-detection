# -*- coding: utf-8 -*-
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers
data = r"D:\work\gygj\任务书数据+代码"
for n in [1, 6]:
    _, d, _ = imgio.load_station(data, n)
    m = imgio.valid_mask(d)
    v = d[m]
    lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
    edges = np.arange(lo, hi + 2, 2)
    hist, _ = np.histogram(v, bins=edges)
    print(f"=== station {n} === range {lo:.0f}-{hi:.0f}mm  valid={int(m.sum())}")
    mx = hist.max()
    for c, e in zip(hist, edges):
        if c > 0:
            bar = "#" * int(c / mx * 60)
            print(f"  {e:7.0f}mm {c:7d} {bar}")
