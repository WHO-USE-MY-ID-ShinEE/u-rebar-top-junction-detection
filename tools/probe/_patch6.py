# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\pipeline.py"
s = io.open(p, encoding="utf-8").read()

s = s.replace("""from pathlib import Path

import cv2
import numpy as np""", """import json
from pathlib import Path

import cv2
import numpy as np""", 1)

old = '''def run_stations(stations, stage="preview", **kwargs):'''
new = '''def export_points(station, result, out_dir=DEFAULT_OUT_DIR):
    """按任务书要求输出交叉点坐标：csv 保存点位集合，json 每行一个 [x, y]。"""
    out_dir = Path(out_dir)
    points = result["accepted"]

    csv_path = out_dir / f"station_{station}_points.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("point_id,x_px,y_px,z_mm\\n")
        for i, (x, y, z) in enumerate(points, 1):
            f.write(f"A{i:03d},{x},{y},{z:.2f}\\n")

    json_path = out_dir / f"station_{station}_points.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"station": station,
                   "size_wh": [int(result["h_lines"][0]["point"].shape[0]) if False else 0],
                   "points": [[int(x), int(y)] for x, y, _ in points]},
                  f, ensure_ascii=False, indent=1)

    # 被剔除的两类点也留档，便于报告里做对比分析
    rej_path = out_dir / f"station_{station}_rejected.json"
    with open(rej_path, "w", encoding="utf-8") as f:
        json.dump({"lower": [[int(x), int(y)] for x, y, _ in result["rejected_lower"]],
                   "clutter": [[int(x), int(y)] for x, y, _ in result["rejected_clutter"]]},
                  f, ensure_ascii=False, indent=1)
    return csv_path, json_path, rej_path


def run_stations(stations, stage="preview", **kwargs):'''
assert old in s
s = s.replace(old, new, 1)

old2 = '''    out_path = out_dir / f"station_{station}_detect.png"
    imgio.imwrite_any(out_path, vis)
    return {"station": station, "out": str(out_path), "stats": res["stats"],
            "split_mm": res["split_mm"]}'''
new2 = '''    out_path = out_dir / f"station_{station}_detect.png"
    imgio.imwrite_any(out_path, vis)
    csv_path, json_path, rej_path = export_points(station, res, out_dir)
    return {"station": station, "out": str(out_path), "csv": str(csv_path),
            "json": str(json_path), "rejected": str(rej_path),
            "stats": res["stats"], "split_mm": res["split_mm"]}'''
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("pipeline patched")
