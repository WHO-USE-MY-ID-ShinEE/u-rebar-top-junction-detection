# -*- coding: utf-8 -*-
"""对比三种顶层深度模型：
  A: depth     = a*x + b*y + c
  B: 1/depth   = a*x + b*y + c      <- 透视投影下平面的正确形式
  C: depth     = 二次曲面
判定指标：顶层像素的残差 p2~p98 跨度（越小越好）。
"""
import sys, numpy as np
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"


def fit(A, z, iters=8):
    keep = np.ones(z.shape, bool)
    coef = None
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        nk = np.abs(r - np.median(r)) <= max(3 * mad, 1e-9)
        if nk.sum() < 0.3 * z.size or np.array_equal(nk, keep):
            break
        keep = nk
    r = z - A @ coef
    return coef, r


def span(r):
    return np.percentile(r, 98) - np.percentile(r, 2)


print(f"{'站':>3} | {'A: depth~xy':>26} | {'B: 1/depth~xy':>26} | {'C: depth~二次':>26}")
print(f"{'':>3} | {'跨度mm':>8} {'MAD':>8} {'R2':>8} | {'跨度mm':>8} {'MAD':>8} {'R2':>8} "
      f"| {'跨度mm':>8} {'MAD':>8} {'R2':>8}")
for n in range(1, 11):
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    near = m & (d < split)
    y, x = np.nonzero(near)
    z = d[near].astype(np.float64)
    xf, yf = x.astype(np.float64), y.astype(np.float64)
    xc, yc = xf - xf.mean(), yf - yf.mean()          # 去中心化，避免病态

    A1 = np.column_stack([xc, yc, np.ones_like(xc)])
    A2 = np.column_stack([xc, yc, np.ones_like(xc)])
    A3 = np.column_stack([xc, yc, xc*xc, yc*yc, xc*yc, np.ones_like(xc)])

    c1, r1 = fit(A1, z)
    inv = 1.0 / z
    c2, r2i = fit(A2, inv)
    r2 = (1.0 / (inv - r2i)) - z                    # 换算回 mm 残差
    c3, r3 = fit(A3, z)

    def r2score(r, z):
        return 1.0 - np.var(r) / np.var(z)

    print(f"{n:>3} | {span(r1):>8.1f} {1.4826*np.median(np.abs(r1-np.median(r1))):>8.2f} "
          f"{r2score(r1,z):>8.3f} | {span(r2):>8.1f} "
          f"{1.4826*np.median(np.abs(r2-np.median(r2))):>8.2f} {r2score(r2,z):>8.3f} "
          f"| {span(r3):>8.1f} {1.4826*np.median(np.abs(r3-np.median(r3))):>8.2f} "
          f"{r2score(r3,z):>8.3f}")
