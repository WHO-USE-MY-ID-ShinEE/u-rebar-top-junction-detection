# -*- coding: utf-8 -*-
"""测量横/竖筋连通域的真实走向与有效筋条数（滤掉开运算产生的小碎块）。"""
import sys, numpy as np, cv2
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"
kh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))


def bars(mask, min_area=3000):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(lab == i)
        out.append((xs, ys))
    return out


for n in [1, 4, 7, 10]:
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    top = (m & (d < layers.auto_split_mm(d))).astype(np.uint8)
    tc = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    H = bars(cv2.morphologyEx(tc, cv2.MORPH_OPEN, kh))
    V = bars(cv2.morphologyEx(tc, cv2.MORPH_OPEN, kv))
    print(f"===== station {n}: 有效横筋 {len(H)} 条, 有效竖筋 {len(V)} 条 =====")
    for tag, bl in [("H", H), ("V", V)]:
        rows = []
        for xs, ys in bl:
            if tag == "H":
                u = np.unique(xs)
                prof = np.array([np.median(ys[xs == uu]) for uu in u])
            else:
                u = np.unique(ys)
                prof = np.array([np.median(xs[ys == uu]) for uu in u])
            A = np.column_stack([u.astype(float), np.ones(u.size)])
            coef, *_ = np.linalg.lstsq(A, prof, rcond=None)
            res = prof - A @ coef
            rows.append((u.min(), u.max(), float(coef[0]), float(np.abs(res).max()),
                         float(res.std())))
        rows.sort(key=lambda r: r[0])
        for r in rows:
            print(f"  {tag} u∈[{r[0]:5.0f},{r[1]:5.0f}] 斜率={r[2]:+.4f} "
                  f"({np.degrees(np.arctan(r[2])):+.2f}°) 最大偏差={r[3]:5.1f}px 标准差={r[4]:5.1f}px")
