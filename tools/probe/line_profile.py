"""沿一条筋线打印"该处是否真有掩膜"的剖面，判断某条筋线是真筋还是外推出来的。

用法：python tools/probe/line_profile.py 站点号 H|V 斜率 截距 [步长px]
  每格代表 step 像素；# = 顶层有掩膜，o = 只有下层，. = 什么都没有
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, layers, pipeline  # noqa: E402


def main(argv):
    st, fam, a, b = int(argv[0]), argv[1].upper(), float(argv[2]), float(argv[3])
    step = int(argv[4]) if len(argv) > 4 else 10
    gray, depth_mm, _ = imgio.load_station(pipeline.DEFAULT_DATA_DIR, st)
    split = layers.auto_split_mm(depth_mm)
    top, far, _ = layers.split_layers(depth_mm, split)
    h, w = depth_mm.shape
    half = 13

    lo, hi = (0, h - 1) if fam == "V" else (0, w - 1)
    marks = []
    for t in range(lo, hi + 1, step):
        if fam == "V":
            x, y = int(round(a * t + b)), t
        else:
            x, y = t, int(round(a * t + b))
        if not (0 <= x < w and 0 <= y < h):
            marks.append(" ")
            continue
        if fam == "V":
            sl = top[y, max(0, x - half):min(w, x + half + 1)]
            fl = far[y, max(0, x - half):min(w, x + half + 1)]
        else:
            sl = top[max(0, y - half):min(h, y + half + 1), x]
            fl = far[max(0, y - half):min(h, y + half + 1), x]
        marks.append("#" if sl.any() else ("o" if fl.any() else "."))
    print(f"station_{st}  {fam} a={a:+.4f} b={b:.1f}  每格={step}px  "
          f"起点={lo} 终点={hi}")
    for i in range(0, len(marks), 120):
        print(f"{lo + i * step:>5} |" + "".join(marks[i:i + 120]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
