# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\work\gygj")
from src import pipeline
print(f"{'站':>3} {'分界mm':>7} {'横':>3} {'竖':>3} {'保留':>5} {'下层':>5} {'杂物':>5}  可疑合并")
for n in range(1, 11):
    r = pipeline.run_detect(n)
    s = r["stats"]
    print(f"{n:>3} {r['split_mm']:>7.1f} {s['h_bars']:>3} {s['v_bars']:>3} "
          f"{s['accepted']:>5} {s['lower']:>5} {s['clutter']:>5}  {s['suspicious_merged']}")
