# -*- coding: utf-8 -*-
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, detect, layers, imgio as io2

DATA = r"D:\work\gygj\任务书数据+代码"
gray, d, raw = imgio.load_station(DATA, 1)
m = imgio.valid_mask(d)
split = layers.auto_split_mm(d)
top, far, _ = layers.split_layers(d, split)

for name, mask in (("顶层", top), ("下层", far)):
    lines, skel = detect.extract_bar_lines(mask)
    h, v = detect.split_directions(lines)
    print(f"===== {name}: 原始筋条 {len(lines)} 条 -> 横 {len(h)} 竖 {len(v)} =====")
    for tag, group in (("H", h), ("V", v)):
        group = sorted(group, key=lambda l: l["rho"])
        print(f"  --{tag}--")
        for l in group:
            e0, e1 = l["extent"]
            print(f"    theta={np.degrees(l['theta']):+7.2f}° rho={l['rho']:8.1f} "
                  f"span={l['span']:7.1f} 支撑={l['n_support']:6d} 直线偏差={l['straightness']:5.1f}px "
                  f"端点=({e0[0]:.0f},{e0[1]:.0f})->({e1[0]:.0f},{e1[1]:.0f})")
