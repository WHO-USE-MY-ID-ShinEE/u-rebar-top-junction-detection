"""量化两个"是不是实心杂物"的判据在全部保留点上的分布，用于定阈值。

判据 A：四角填充率 —— 真交叉四周是空的，压在实心块上则四角被填满。
判据 B：交叉点局部深度 与 两条筋自身深度 的差 —— 杂物挡在钢筋笼前面时，
        交叉点处的深度会明显比两条筋本身更近。

用法：python tools/probe/calibrate_clutter.py [站点号 ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, pipeline  # noqa: E402

INNER, OUTER = 18, 30


def corner_fill(mask, x, y):
    h, w = mask.shape
    fills = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = sorted((x + sx * INNER, x + sx * OUTER))
            y0, y1 = sorted((y + sy * INNER, y + sy * OUTER))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w), min(y1, h)
            if x1 > x0 and y1 > y0:
                fills.append(float(mask[y0:y1, x0:x1].mean()))
    return float(np.mean(fills)) if fills else None


def line_depth(depth_mm, line, exclude_r=70, reach=200):
    h, w = depth_mm.shape
    vals = []
    for s in np.arange(-reach, reach + 1, 3.0):
        if abs(s) < exclude_r:
            continue
        q = line["point"] + line["direction"] * s
        x, y = int(round(q[0])), int(round(q[1]))
        if 0 <= x < w and 0 <= y < h:
            v = depth_mm[y, x]
            if np.isfinite(v) and v > 0:
                vals.append(float(v))
    return float(np.median(vals)) if len(vals) >= 10 else None


def main(argv):
    data_dir = pipeline.DEFAULT_DATA_DIR
    stations = [int(s) for s in argv] or imgio.list_stations(data_dir)
    allrows = []
    for st in stations:
        gray, depth_mm, _ = imgio.load_station(data_dir, st)
        res = detect.detect_rebar_intersections(depth_mm, gray)
        mask = res["top_mask"]
        for x, y, dz in res["accepted"]:
            cf = corner_fill(mask, x, y)
            # 找到给出该点的两条筋
            best = None
            for a in res["h_lines"]:
                for b in res["v_lines"]:
                    pt = detect.intersect(a, b)
                    if pt is None:
                        continue
                    d = float(np.hypot(pt[0] - x, pt[1] - y))
                    if best is None or d < best[0]:
                        best = (d, a, b)
            gap = None
            if best is not None:
                dh = line_depth(depth_mm, best[1])
                dv = line_depth(depth_mm, best[2])
                ref = [v for v in (dh, dv) if v is not None]
                if ref:
                    gap = float(dz) - min(ref)
            allrows.append((st, x, y, cf, gap))
    print(f"保留点共 {len(allrows)} 个")
    cfs = np.array([r[3] for r in allrows if r[3] is not None])
    print(f"四角填充率：中位={np.median(cfs):.2f} "
          f"p90={np.percentile(cfs, 90):.2f} p99={np.percentile(cfs, 99):.2f} "
          f"最大={cfs.max():.2f}")
    gaps = np.array([r[4] for r in allrows if r[4] is not None])
    print(f"交叉点比筋条更近的量：中位={np.median(gaps):+.1f}mm "
          f"p1={np.percentile(gaps, 1):+.1f}mm 最小={gaps.min():+.1f}mm")
    print("\n四角填充率最大的 8 个保留点：")
    for r in sorted(allrows, key=lambda z: -(z[3] if z[3] is not None else -1))[:8]:
        print(f"  station_{r[0]:<2} ({r[1]:>4},{r[2]:>4})  四角={r[3]:.2f}  "
              f"深度差={r[4]:+.1f}mm")
    print("\n交叉点比筋条更近最多的 8 个保留点：")
    for r in sorted(allrows, key=lambda z: (z[4] if z[4] is not None else 1e9))[:8]:
        print(f"  station_{r[0]:<2} ({r[1]:>4},{r[2]:>4})  四角={r[3]:.2f}  "
              f"深度差={r[4]:+.1f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
