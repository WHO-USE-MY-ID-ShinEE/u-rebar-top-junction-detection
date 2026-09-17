# -*- coding: utf-8 -*-
"""量化检查：顶层掩膜里是否混入了下层像素（泄漏）。

参考面用 1/depth = a*x + b*y + c 拟合顶层（透视投影下平面的正确形式），
然后看下层像素相对该参考面的残差有多大 —— 残差越大说明深度分离越可靠，
顶层掩膜里一旦混入下层像素，其残差会明显偏正、形成长尾。
"""
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"


def fit_inv_plane(d, mask, iters=8):
    y, x = np.nonzero(mask)
    z = d[mask].astype(np.float64)
    xc, yc = (x - x.mean()).astype(np.float64), (y - y.mean()).astype(np.float64)
    A = np.column_stack([xc, yc, np.ones_like(xc)])
    keep = np.ones(z.shape, bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], 1.0 / z[keep], rcond=None)
        pred = 1.0 / (A @ coef)
        r = z - pred
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        nk = np.abs(r - np.median(r)) <= max(3 * mad, 1e-6)
        if nk.sum() < 0.3 * z.size or np.array_equal(nk, keep):
            break
        keep = nk
    # 返回全图残差函数
    def residual(depth_img):
        h, w = depth_img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        xf, yf = (xx - x.mean()).astype(np.float64), (yy - y.mean()).astype(np.float64)
        Aall = np.column_stack([xf.ravel(), yf.ravel(), np.ones(xx.size)])
        return depth_img.astype(np.float64) - (1.0 / (Aall @ coef)).reshape(h, w)
    return residual


print(f"{'站':>3} | {'顶层残差p2':>9} {'p50':>7} {'p98':>7} {'p99.9':>7} | "
      f"{'下层残差p2':>10} {'p50':>7} | {'顶层泄漏率':>9} {'分离裕度mm':>10}")
for n in range(1, 11):
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    near = m & (d < split)
    far = m & (d >= split)
    res = fit_inv_plane(d, near)(d)
    rn, rf = res[near], res[far]
    # 泄漏率：顶层像素中残差超过 60mm 的比例
    leak = float((rn > 60.0).mean())
    margin = float(np.percentile(rf, 2) - np.percentile(rn, 98))
    print(f"{n:>3} | {np.percentile(rn,2):>9.1f} {np.median(rn):>7.1f} "
          f"{np.percentile(rn,98):>7.1f} {np.percentile(rn,99.9):>7.1f} | "
          f"{np.percentile(rf,2):>10.1f} {np.median(rf):>7.1f} | "
          f"{leak*100:>8.3f}% {margin:>10.1f}")
