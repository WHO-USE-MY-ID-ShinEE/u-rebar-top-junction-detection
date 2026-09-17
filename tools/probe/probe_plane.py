# -*- coding: utf-8 -*-
"""路线图第 1 步验证：顶层深度平面拟合。

问题：顶层是一个平面网格，但相机斜视，同一层内深度跨度可达约 88mm，
      而层间空段只有 22~96mm。两者量级相当 -> 全局阈值在原理上就是脆弱的。
思路：把顶层深度拟合成平面 depth = a*x + b*y + c，用残差代替绝对深度做判定。
"""
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"


def fit_plane(depth, mask, iters=8):
    y, x = np.nonzero(mask)
    z = depth[mask].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)]).astype(np.float64)
    keep = np.ones(z.shape, bool)
    coef = None
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med))
        tol = max(3.0 * mad, 0.5)
        nk = np.abs(r - med) <= tol
        if nk.sum() < 0.3 * z.size or np.array_equal(nk, keep):
            break
        keep = nk
    r = z - A @ coef
    res_keep = r[keep]
    med = np.median(res_keep)
    mad = 1.4826 * np.median(np.abs(res_keep - med))
    return coef, med, mad, keep.mean(), r, A


print(f"{'站':>3} {'分界mm':>7} {'顶层跨度mm':>9} | {'残差MAD':>7} {'残差p2~p98':>16} "
      f"| {'远处层残差中位':>13} {'分离倍数':>7}")
results = {}
for n in range(1, 11):
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    near = m & (d < split)
    far = m & (d >= split)
    coef, med, mad, ratio, r, A = fit_plane(d, near)
    near_span = float(near[near].size and (np.percentile(d[near], 98) - np.percentile(d[near], 2)))
    # 平面预测在整个图上的残差
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pred = coef[0] * xx + coef[1] * yy + coef[2]
    res_all = d - pred
    far_res = res_all[far] if far.any() else np.array([np.nan])
    sep = float(np.median(far_res) / (3 * mad)) if far.any() else float("nan")
    results[n] = (coef, med, mad, ratio, res_all, m, near, far)
    print(f"{n:>3} {split:>7.1f} {near_span:>9.1f} | {mad:>7.2f} "
          f"{np.percentile(r,2):>7.1f}~{np.percentile(r,98):<8.1f}"
          f"| {np.median(far_res):>13.1f} {sep:>7.2f}")

np.save(r"D:\work\gygj\tools\probe\_plane_cache.npy",
        np.array([results[n][0] for n in range(1, 11)]))
print("\n平面系数已缓存。残差 = depth - (a*x + b*y + c)，单位 mm")
