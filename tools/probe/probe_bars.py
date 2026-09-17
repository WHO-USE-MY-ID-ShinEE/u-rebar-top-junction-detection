# -*- coding: utf-8 -*-
"""为"定向开运算 + 连通域 + PCA 拟合筋轴线"摸参数：
统计每站顶层掩膜经横/竖开运算后的连通域数量、面积、长度。
同时确认 cv2.ximgproc（骨架化）是否可用。
"""
import sys, numpy as np, cv2
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

print("cv2.ximgproc 可用:", hasattr(cv2, "ximgproc"))
print("cv2 版本:", cv2.__version__)

DATA = r"D:\work\gygj\任务书数据+代码"
K = 25
kh = cv2.getStructuringElement(cv2.MORPH_RECT, (K, 1))
kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, K))


def comps(mask, min_area=200):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        t = (pts - c) @ d
        # 到主轴的最大垂直距离，衡量直不直
        dev = float(np.abs((pts - c) @ vt[1]).max()) if vt.shape[0] > 1 else 0.0
        out.append({"area": int(stats[i, cv2.CC_STAT_AREA]), "len": float(t.max() - t.min()),
                    "cx": float(c[0]), "cy": float(c[1]),
                    "angle": float(np.degrees(np.arctan2(d[1], d[0]))),
                    "dev": dev,
                    "bbox": tuple(int(v) for v in stats[i, :4])})
    return out


for n in range(1, 11):
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    top = (m & (d < split)).astype(np.uint8)
    top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    H = comps(cv2.morphologyEx(top, cv2.MORPH_OPEN, kh))
    V = comps(cv2.morphologyEx(top, cv2.MORPH_OPEN, kv))
    hl = sorted(c["len"] for c in H)[-3:]
    vl = sorted(c["len"] for c in V)[-3:]
    print(f"st{n:>2}: 横筋连通域 {len(H):>2} 个 最长3={[int(x) for x in hl]} "
          f"角度中位={np.median([c['angle'] for c in H]):>6.1f} "
          f"直线偏差中位={np.median([c['dev'] for c in H]):>5.1f}px | "
          f"竖筋 {len(V):>2} 个 最长3={[int(x) for x in vl]} "
          f"角度中位={np.median([c['angle'] for c in V]):>6.1f} "
          f"直线偏差中位={np.median([c['dev'] for c in V]):>5.1f}px")
