# -*- coding: utf-8 -*-
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"
for n in [4, 7]:
    gray, d, raw = imgio.load_station(DATA, n)
    split = layers.auto_split_mm(d)
    top, far, _ = layers.split_layers(d, split)
    lines, skel = detect.extract_bar_lines(top)
    h, v = detect.split_directions(lines)
    print(f"===== station {n}: H={len(h)} V={len(v)} 骨架像素={int(skel.sum())} =====")
    for tag, g in (("H", h), ("V", v)):
        for l in sorted(g, key=lambda z: z["ab"][1]):
            print(f"  {tag} a={l['ab'][0]:+8.4f} b={l['ab'][1]:8.1f} "
                  f"span={l['span']:7.1f} 支撑={l['n_support']:5d} "
                  f"支撑/跨度={l['coverage']:5.2f} 直线偏差={l['straightness']:5.1f}px")
