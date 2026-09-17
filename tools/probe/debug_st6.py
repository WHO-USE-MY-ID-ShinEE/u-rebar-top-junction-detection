# -*- coding: utf-8 -*-
"""诊断 station_6 漏检的竖筋：分别看顶层/下层掩膜检出的筋条，
以及被 min_coverage 滤掉的候选线。"""
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"
n = 6
gray, d, raw = imgio.load_station(DATA, n)
split = layers.auto_split_mm(d)
top, far, _ = layers.split_layers(d, split)
print(f"station {n}: 分界 {split:.1f}mm  顶层像素 {int(top.sum())}  下层像素 {int(far.sum())}")

p = dict(detect.DEFAULTS)
p_far = dict(p)
p_far["hough_min_len"] = p["far_hough_min_len"]
p_far["hough_thresh"] = p["far_hough_thresh"]

for name, mask, pp in (("顶层", top, p), ("下层", far, p_far)):
    lines, skel = detect.extract_bar_lines(mask, pp)
    h, v = detect.split_directions(lines)
    print(f"\n===== {name}掩膜: 骨架 {int(skel.sum())} 像素 -> H={len(h)} V={len(v)} =====")
    for tag, g in (("H", h), ("V", v)):
        for l in sorted(g, key=lambda z: z["ab"][1]):
            print(f"  {tag} a={l['ab'][0]:+7.4f} b={l['ab'][1]:8.1f} span={l['span']:7.1f} "
                  f"支撑={l['n_support']:5d} 覆盖={l['coverage']:5.2f}")

# 关掉覆盖度过滤，看下层到底有多少候选竖筋
p_loose = dict(p_far); p_loose["min_coverage"] = 0.0; p_loose["min_support_px"] = 40
lines, _ = detect.extract_bar_lines(far, p_loose)
h, v = detect.split_directions(lines)
print(f"\n===== 下层（放宽过滤）: H={len(h)} V={len(v)} =====")
for l in sorted(v, key=lambda z: z["ab"][1]):
    print(f"  V a={l['ab'][0]:+7.4f} b={l['ab'][1]:8.1f} span={l['span']:7.1f} "
          f"支撑={l['n_support']:5d} 覆盖={l['coverage']:5.2f}")
