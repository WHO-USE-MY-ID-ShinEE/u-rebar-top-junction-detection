# -*- coding: utf-8 -*-
"""检验假设：顶层网格里横筋与竖筋是否处在不同深度。
方法：用定向形态学开运算把顶层掩膜拆成"横筋像素"和"竖筋像素"，
      比较两者的深度分布，并对各自单独拟合平面。
"""
import sys, numpy as np, cv2
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"
K = 25


def fit(A, z, iters=8):
    keep = np.ones(z.shape, bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        nk = np.abs(r - np.median(r)) <= max(3 * mad, 1e-9)
        if nk.sum() < 0.3 * z.size or np.array_equal(nk, keep):
            break
        keep = nk
    return coef, z - A @ coef


print(f"{'站':>3} | {'横筋中位':>8} {'竖筋中位':>8} {'差':>7} | "
      f"{'横筋残差跨度':>12} {'竖筋残差跨度':>12} | {'两者合并跨度':>12}")
kh = cv2.getStructuringElement(cv2.MORPH_RECT, (K, 1))
kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, K))
for n in range(1, 11):
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    near = (m & (d < split)).astype(np.uint8)
    hm = cv2.morphologyEx(near, cv2.MORPH_OPEN, kh) > 0
    vm = cv2.morphologyEx(near, cv2.MORPH_OPEN, kv) > 0
    hm_only = hm & ~vm
    vm_only = vm & ~hm
    dh = np.median(d[hm_only]); dv = np.median(d[vm_only])
    spans = []
    for mask in (hm_only, vm_only, hm | vm):
        y, x = np.nonzero(mask)
        z = d[mask].astype(np.float64)
        xc, yc = (x - x.mean()).astype(np.float64), (y - y.mean()).astype(np.float64)
        A = np.column_stack([xc, yc, np.ones_like(xc)])
        _, r = fit(A, z)
        spans.append(np.percentile(r, 98) - np.percentile(r, 2))
    print(f"{n:>3} | {dh:>8.1f} {dv:>8.1f} {dv-dh:>7.1f} | "
          f"{spans[0]:>12.1f} {spans[1]:>12.1f} | {spans[2]:>12.1f}")
