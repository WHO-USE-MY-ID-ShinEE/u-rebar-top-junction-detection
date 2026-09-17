# -*- coding: utf-8 -*-
"""看下层斜率容许量的敏感性，选定参数后收尾。"""
import sys, importlib
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"
for spread in (0.02, 0.03, 0.04):
    print(f"--- far_slope_spread = {spread} ---")
    print(f"{'站':>3} {'下层H':>5} {'下层V':>5} {'下层点':>6}")
    for n in range(1, 11):
        gray, d, raw = imgio.load_station(DATA, n)
        res = detect.detect_rebar_intersections(d, gray, {"far_slope_spread": spread})
        s = res["stats"]
        print(f"{n:>3} {s['far_h_bars']:>5} {s['far_v_bars']:>5} {s['lower']:>6}")
