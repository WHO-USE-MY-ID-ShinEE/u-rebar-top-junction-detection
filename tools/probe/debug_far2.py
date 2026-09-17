# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"
for n in (1, 2):
    gray, d, raw = imgio.load_station(DATA, n)
    split = layers.auto_split_mm(d)
    top, far, _ = layers.split_layers(d, split)
    p = dict(detect.DEFAULTS)
    lp = dict(p); lp["hough_min_len"] = p["far_hough_min_len"]
    lp["hough_thresh"] = p["far_hough_thresh"]
    lp["min_coverage"] = p["far_min_coverage"]
    lp["min_support_px"] = p["far_min_support_px"]
    tl, _ = detect.extract_bar_lines(top, p)
    _, tv = detect.split_directions(tl)
    print(f"===== station {n} 顶层竖筋斜率: "
          f"{sorted(round(l['ab'][0], 4) for l in tv)}")
    fl, _ = detect.extract_bar_lines(far, lp)
    _, fv = detect.split_directions(fl)
    print(f"  下层竖筋（未加斜率约束）{len(fv)} 条:")
    for l in sorted(fv, key=lambda z: z["ab"][1]):
        print(f"    a={l['ab'][0]:+7.4f} b={l['ab'][1]:8.1f} span={l['span']:7.1f} "
              f"支撑={l['n_support']:5d} 覆盖={l['coverage']:5.2f}")
