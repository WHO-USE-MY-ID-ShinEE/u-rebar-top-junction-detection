"""解释某个可疑点：找出给出该交点的两条筋线，打印它们的参数、跨度与端点。

用法：python tools/probe/explain_point.py 站点号 x y
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, pipeline  # noqa: E402


def _fmt(l):
    a, b = l["ab"]
    ex = l["extent"]
    return (f"{l['family']} a={a:+.4f} b={b:7.1f} span={l['span']:6.1f} "
            f"cov={l['coverage']:.2f} n={l['n_support']:5d} "
            f"端点到端: ({ex[0][0]:6.1f},{ex[0][1]:6.1f}) -> "
            f"({ex[1][0]:6.1f},{ex[1][1]:6.1f})")


def main(argv):
    st, x, y = int(argv[0]), int(argv[1]), int(argv[2])
    gray, depth_mm, _ = imgio.load_station(pipeline.DEFAULT_DATA_DIR, st)
    res = detect.detect_rebar_intersections(depth_mm, gray)
    p = detect.DEFAULTS
    best = None
    for a in res["h_lines"]:
        for b in res["v_lines"]:
            pt = detect.intersect(a, b)
            if pt is None:
                continue
            d = float(np.hypot(pt[0] - x, pt[1] - y))
            if best is None or d < best[0]:
                best = (d, a, b, pt)
    d, a, b, pt = best
    print(f"station_{st} 目标点=({x},{y})  最近交点=({pt[0]:.1f},{pt[1]:.1f}) "
          f"距离={d:.1f}px")
    print(f"  横筋线: {_fmt(a)}")
    print(f"  竖筋线: {_fmt(b)}")
    for tag, line in (("横", a), ("竖", b)):
        pos, neg = detect._direction_support(res["top_mask"], pt, line, p)
        print(f"  {tag}筋两向支撑比: 正={pos}  负={neg}")
    print(f"  就地判定: {detect.crossing_supported(res['top_mask'], pt, a, b, p)}")
    print(f"  局部深度: {detect.local_depth(depth_mm, int(round(pt[0])), int(round(pt[1])))}")
    print(f"  深度分界: {res['split_mm']:.1f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
