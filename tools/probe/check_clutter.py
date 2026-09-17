"""检查非钢筋杂物（灰度图上的亮色块）附近有没有被误当成交叉点。

灰度图整体极暗（均值约 17），非钢筋杂物（白色圆形垫块、浅色长条）在灰度上明显
偏亮，因此可用高阈值连通域把它们找出来，再看检测结果里有没有点落在它们身上。

用法：python tools/probe/check_clutter.py [站点号 ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, pipeline  # noqa: E402

MIN_AREA = 120


def main(argv):
    data_dir = pipeline.DEFAULT_DATA_DIR
    stations = [int(s) for s in argv] or imgio.list_stations(data_dir)
    for st in stations:
        gray, depth_mm, _ = imgio.load_station(data_dir, st)
        res = detect.detect_rebar_intersections(depth_mm, gray)
        acc = [(x, y) for x, y, _ in res["accepted"]]
        bright = (gray > 60).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
        blobs = [(int(stats[i, cv2.CC_STAT_AREA]), float(cent[i][0]),
                  float(cent[i][1])) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
        blobs.sort(reverse=True)
        hit = []
        for area, cx, cy in blobs:
            if not acc:
                break
            d = min(np.hypot(px - cx, py - cy) for px, py in acc)
            if d < 60:
                hit.append((area, cx, cy, d))
        print(f"station_{st}: 亮色块 {len(blobs)} 个"
              f"（最大面积 {blobs[0][0] if blobs else 0}），"
              f"其中被检出的交叉点压住的 {len(hit)} 个")
        for area, cx, cy, d in hit[:5]:
            print(f"    \u8d28\u5fc3=({cx:.0f},{cy:.0f}) \u9762\u79ef={area} "
                  f"\u6700\u8fd1\u68c0\u51fa\u70b9\u8ddd\u79bb={d:.1f}px")
        for area, cx, cy in blobs[:3]:
            print(f"    \u4eae\u5757: \u8d28\u5fc3=({cx:.0f},{cy:.0f}) \u9762\u79ef={area}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
