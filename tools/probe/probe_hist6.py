# -*- coding: utf-8 -*-
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio
data = r"D:\work\gygj\任务书数据+代码"
_, d, _ = imgio.load_station(data, 6)
m = imgio.valid_mask(d); v = d[m]
edges = np.arange(800, 1200, 2)
hist, _ = np.histogram(v, bins=edges)
mx = hist.max()
for c, e in zip(hist, edges):
    print(f"  {e:6.0f}mm {c:6d} " + ("#" * int(c / mx * 55)))
print("更深像素 (>1200mm) 占比: %.4f" % ((v > 1200).mean()))
