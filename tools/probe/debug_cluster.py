# -*- coding: utf-8 -*-
import sys, numpy as np, cv2
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"
p = dict(detect.DEFAULTS)

for n in [1, 7]:
    gray, d, raw = imgio.load_station(DATA, n)
    split = layers.auto_split_mm(d)
    top, far, _ = layers.split_layers(d, split)
    closed = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((p["close_ksize"],) * 2, np.uint8))
    skel = detect.zhang_suen(closed, p["skel_max_iter"])
    segs = detect._hough_segments(skel, p)
    reps = {"H": [], "V": []}
    for s in segs:
        r = detect._segment_ab(s)
        if r:
            reps[r[0]].append((r[1], r[2]))
    print(f"===== station {n}: 骨架像素 {int(skel.sum())}, Hough线段 {len(segs)} "
          f"(H族 {len(reps['H'])}, V族 {len(reps['V'])}) =====")
    for fam in ("H", "V"):
        cl = detect._cluster_ab(reps[fam], p)
        print(f"  {fam}族 聚类数={len(cl)}")
        ys, xs = np.nonzero(skel)
        xs, ys = xs.astype(float), ys.astype(float)
        for a, b, k in sorted(cl, key=lambda z: z[1]):
            dist = detect._point_to_line_dist(xs, ys, fam, a, b)
            sel = int((dist < p["refit_halfwidth"]).sum())
            flag = "" if sel >= p["min_support_px"] else "  <- 支撑不足被丢弃"
            print(f"    a={a:+8.4f} b={b:8.1f} 段数={k:3d} 邻近骨架像素={sel:5d}{flag}")
