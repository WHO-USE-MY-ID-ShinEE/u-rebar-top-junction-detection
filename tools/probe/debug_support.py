"""诊断：打印某个交叉点在两条筋方向上的逐点取样结果。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, pipeline  # noqa: E402

st, x, y = 9, 1049, 1397
gray, depth_mm, _ = imgio.load_station(pipeline.DEFAULT_DATA_DIR, st)
res = detect.detect_rebar_intersections(depth_mm, gray)
p = detect.DEFAULTS
mask = res["top_mask"]
best = None
for a in res["h_lines"]:
    for b in res["v_lines"]:
        pt = detect.intersect(a, b)
        d = float(np.hypot(pt[0] - x, pt[1] - y))
        if best is None or d < best[0]:
            best = (d, a, b, pt)
_, A, B, pt = best
for tag, line in (("H", A), ("V", B)):
    dirv = line["direction"]
    proj = [(e - line["point"]) @ dirv for e in line["extent"]]
    lo, hi = min(proj), max(proj)
    off = float((pt - line["point"]) @ dirv)
    print(f"{tag} 筋: 中心=({line['point'][0]:.1f},{line['point'][1]:.1f}) "
          f"方向=({dirv[0]:+.4f},{dirv[1]:+.4f}) 跨度={line['span']:.1f}")
    print(f"   交点在轴上的位置={off:+.1f}   实测跨度区间=[{lo:+.1f}, {hi:+.1f}]")
    for sign in (1.0, -1.0):
        cells = []
        for t in np.linspace(10.0, 50.0, 9):
            s = sign * t
            q = pt + dirv * s
            xi, yi = int(round(q[0])), int(round(q[1]))
            if not lo <= s <= hi:
                cells.append("跳")
                continue
            if not (0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]):
                cells.append("越")
                continue
            x0, x1 = max(0, xi - 12), min(mask.shape[1], xi + 13)
            y0, y1 = max(0, yi - 12), min(mask.shape[0], yi + 13)
            cells.append("#" if mask[y0:y1, x0:x1].any() else ".")
        print(f"   方向 {sign:+.0f}: " + " ".join(cells))
    print()
