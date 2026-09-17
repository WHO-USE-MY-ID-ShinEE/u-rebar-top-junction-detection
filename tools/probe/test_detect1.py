# -*- coding: utf-8 -*-
import sys, time, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, detect

DATA = r"D:\work\gygj\任务书数据+代码"
t0 = time.time()
gray, d, raw = imgio.load_station(DATA, 1)
print(f"载入 {time.time()-t0:.1f}s")
t0 = time.time()
res = detect.detect_rebar_intersections(d, gray)
print(f"检测 {time.time()-t0:.1f}s")
print("分层阈值:", round(res["split_mm"], 1), "mm")
print("顶层筋条: 横", len(res["h_lines"]), " 竖", len(res["v_lines"]))
print("统计:", res["stats"])
print()
print("保留的交叉点 (x, y, 深度mm):")
for i, (x, y, z) in enumerate(sorted(res["accepted"], key=lambda t: (t[1], t[0]))):
    print(f"  P{i+1:02d} ({x:4d},{y:4d}) z={z:6.1f}")
