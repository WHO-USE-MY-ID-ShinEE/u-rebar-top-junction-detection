# -*- coding: utf-8 -*-
"""查 station_1 横筋连通域为何过多。"""
import sys, numpy as np, cv2
sys.path.insert(0, r"D:\work\gygj")
from src import imgio, layers

DATA = r"D:\work\gygj\任务书数据+代码"
kh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

for n in [1, 7]:
    gray, d, raw = imgio.load_station(DATA, n)
    m = imgio.valid_mask(d)
    split = layers.auto_split_mm(d)
    top = (m & (d < split)).astype(np.uint8)
    print(f"===== station {n} =====")
    print("  顶层掩膜连通域数(原始):", cv2.connectedComponents(top, 8)[0] - 1)
    tc = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    H = cv2.morphologyEx(tc, cv2.MORPH_OPEN, kh)
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(H, 8)
    items = [(int(stats[i, cv2.CC_STAT_AREA]), tuple(int(v) for v in stats[i, :4]))
             for i in range(1, nlab)]
    items.sort(reverse=True)
    print(f"  横筋开运算后连通域数: {len(items)}  总面积 {int(H.sum())}")
    print("  按面积排序的前 20 个 (面积, bbox[x,y,w,h]):")
    for a, b in items[:20]:
        print(f"    {a:7d}  x={b[0]:5d} y={b[1]:5d} w={b[2]:5d} h={b[3]:4d}")
    small = [a for a, _ in items if a < 3000]
    print(f"  面积<3000 的小碎块: {len(small)} 个, 占面积 "
          f"{sum(small)/max(int(H.sum()),1)*100:.1f}%")
