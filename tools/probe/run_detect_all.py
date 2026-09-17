# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\work\gygj")
from src import pipeline, imgio, layers, detect

DATA = r"D:\work\gygj\任务书数据+代码"
print(f"{'站':>3} | {'顶层H':>5} {'顶层V':>5} | {'下层H':>5} {'下层V':>5} | "
      f"{'保留':>5} {'下层点':>6} {'杂物':>5}")
for n in range(1, 11):
    r = pipeline.run_detect(n)
    s = r["stats"]
    print(f"{n:>3} | {s['h_bars']:>5} {s['v_bars']:>5} | {s['far_h_bars']:>5} "
          f"{s['far_v_bars']:>5} | {s['accepted']:>5} {s['lower']:>6} {s['clutter']:>5}")

print()
print("station 6 下层竖筋明细:")
gray, d, raw = imgio.load_station(DATA, 6)
split = layers.auto_split_mm(d)
top, far, _ = layers.split_layers(d, split)
p = dict(detect.DEFAULTS)
p["hough_min_len"] = p["far_hough_min_len"]
p["hough_thresh"] = p["far_hough_thresh"]
p["min_coverage"] = p["far_min_coverage"]
p["min_support_px"] = p["far_min_support_px"]
lines, _ = detect.extract_bar_lines(far, p)
_, v = detect.split_directions(lines)
for l in sorted(v, key=lambda z: z["ab"][1]):
    print(f"  V a={l['ab'][0]:+7.4f} b={l['ab'][1]:8.1f} span={l['span']:7.1f} "
          f"支撑={l['n_support']:5d} 覆盖={l['coverage']:5.2f}")
