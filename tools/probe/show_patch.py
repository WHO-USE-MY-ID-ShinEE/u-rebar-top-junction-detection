"""在终端里用 ASCII 打印某点附近的掩膜，便于在没有看图能力时判断点位真假。

用法：python tools/probe/show_patch.py 站点号 x y [半径x] [半径y]
  # = 顶层像素   o = 下层像素   . = 无回波   X = 目标点位置
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, layers, pipeline  # noqa: E402


def main(argv):
    st, x, y = int(argv[0]), int(argv[1]), int(argv[2])
    hx = int(argv[3]) if len(argv) > 3 else 45
    hy = int(argv[4]) if len(argv) > 4 else 18
    gray, depth_mm, _ = imgio.load_station(pipeline.DEFAULT_DATA_DIR, st)
    split = layers.auto_split_mm(depth_mm)
    top, far, valid = layers.split_layers(depth_mm, split)
    h, w = depth_mm.shape

    res = detect.detect_rebar_intersections(depth_mm, gray)
    acc = [(px, py) for px, py, _ in res["accepted"]]

    print(f"station_{st}  目标点=({x},{y})  深度分界={split:.1f}mm  "
          f"图像={w}x{h}")
    d = detect.local_depth(depth_mm, x, y)
    print(f"该点局部深度：{d}")
    near = [p for p in acc if abs(p[0] - x) <= hx and abs(p[1] - y) <= hy]
    print(f"附近已保留点：{near}")
    print()
    for yy in range(y - hy, y + hy + 1):
        row = []
        for xx in range(x - hx, x + hx + 1):
            if not (0 <= xx < w and 0 <= yy < h):
                row.append(" ")
            elif xx == x and yy == y:
                row.append("X")
            elif top[yy, xx]:
                row.append("#")
            elif far[yy, xx]:
                row.append("o")
            else:
                row.append(".")
        print("".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
