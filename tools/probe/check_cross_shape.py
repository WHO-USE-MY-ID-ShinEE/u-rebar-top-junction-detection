"""检查每个检出点周围的掩膜是不是"十字形"（真交叉），而不是实心块（杂物）。

真钢筋交叉：局部掩膜是两条约一个钢筋直径宽的条带正交，四角区域是空的。
圆形垫块一类的实心杂物：局部掩膜近似一个圆盘，四角会被填上。
因此用"四角填充率"就能把"压在实心块上的点"挑出来。

用法：python tools/probe/check_cross_shape.py [站点号 ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, pipeline  # noqa: E402

OUT = 28      # 窗口半径
INNER = 16    # 四角块的起始偏移（超过钢筋半宽）


def corner_fill(mask, x, y):
    h, w = mask.shape
    fills = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = x + sx * OUT, x + sx * INNER
            y0, y1 = y + sy * OUT, y + sy * INNER
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            patch = mask[y0:y1, x0:x1]
            if patch.size:
                fills.append(float(patch.mean()))
    return float(np.mean(fills)) if fills else None


def main(argv):
    data_dir = pipeline.DEFAULT_DATA_DIR
    stations = [int(s) for s in argv] or imgio.list_stations(data_dir)
    for st in stations:
        gray, depth_mm, _ = imgio.load_station(data_dir, st)
        res = detect.detect_rebar_intersections(depth_mm, gray)
        mask = res["top_mask"]
        rows = []
        for tag, pts in (("保留", res["accepted"]),
                         ("下层", res["rejected_lower"]),
                         ("干扰", res["rejected_clutter"])):
            for x, y, _ in pts:
                f = corner_fill(mask, x, y)
                if f is not None:
                    rows.append((f, tag, x, y))
        rows.sort(reverse=True)
        high = [r for r in rows if r[0] > 0.4]
        print(f"station_{st}: 检出点 {len(rows)} 个，四角填充率最高 "
              f"{rows[0][0]:.2f}（{rows[0][1]} 点 ({rows[0][2]},{rows[0][3]})），"
              f">0.4 的有 {len(high)} 个")
        for f, tag, x, y in high[:6]:
            print(f"    \u56db\u89d2\u586b\u5145\u7387={f:.2f}  {tag}点 ({x},{y})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
