# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"
for n in (1, 2, 3):
    gray, d, raw = imgio.load_station(DATA, n)
    split = layers.auto_split_mm(d)
    top, far, _ = layers.split_layers(d, split)
    p = dict(detect.DEFAULTS)
    p["hough_min_len"] = p["far_hough_min_len"]
    p["hough_thresh"] = p["far_hough_thresh"]
    p["min_coverage"] = p["far_min_coverage"]
    p["min_support_px"] = p["far_min_support_px"]
    lines, skel = detect.extract_bar_lines(far, p)
    h, v = detect.split_directions(lines)
    print(f"===== station {n} 下层: H={len(h)} V={len(v)} 骨架 {int(skel.sum())}px =====")
    for l in sorted(v, key=lambda z: z["ab"][1]):
        print(f"  V a={l['ab'][0]:+7.4f} b={l['ab'][1]:8.1f} span={l['span']:7.1f} "
              f"支撑={l['n_support']:5d} 覆盖={l['coverage']:5.2f} "
              f"直线偏差={l['straightness']:5.1f}px")
