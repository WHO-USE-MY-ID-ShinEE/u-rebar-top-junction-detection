"""流程编排：把各模块串成可运行的单站/批量流程。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from . import detect, enhance, imgio, layers, viz

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "任务书数据+代码"
DEFAULT_OUT_DIR = ROOT / "outputs"

SUMMARY_FIELDS = ["station", "split_mm", "top_h_bars", "top_v_bars",
                  "far_h_bars", "far_v_bars", "accepted", "rejected_lower",
                  "rejected_clutter"]


def _stack(panels, pad_color=(255, 255, 255)):
    """把若干图纵向拼接，右侧用白边补齐。"""
    width = max(p.shape[1] for p in panels)
    rows = []
    for p in panels:
        rows.append(cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1],
                                       cv2.BORDER_CONSTANT, value=pad_color))
        rows.append(np.full((10, width, 3), pad_color, np.uint8))
    return np.vstack(rows[:-1])


def _load_enhanced(data_dir, station):
    gray, depth_mm, _ = imgio.load_station(data_dir, station)
    enhanced, _ = enhance.enhance(gray)
    return enhanced, depth_mm


def run_preview(station, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR,
                split_mm=None):
    """阶段一：读数据 -> 灰度增强 -> 分层 -> 输出对照图。"""
    out_dir = Path(out_dir)
    enhanced, depth_mm = _load_enhanced(data_dir, station)
    depth_vis = enhance.depth_visual(depth_mm)

    stats = layers.layer_stats(depth_mm, split_mm)
    near, far, _ = layers.split_layers(depth_mm, stats["split_mm"])
    overlay = viz.to_bgr(enhanced)
    overlay[near] = (overlay[near] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    overlay[far] = (overlay[far] * 0.4 + np.array([255, 128, 0]) * 0.6).astype(np.uint8)
    cv2.putText(overlay, f"red = top layer (<{stats['split_mm']:.0f}mm)  blue = lower",
                (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

    sheet = _stack([viz.fit_width(p) for p in
                    (viz.side_by_side_gray_depth(enhanced, depth_vis), overlay)])
    out_path = out_dir / f"station_{station}_preview.png"
    imgio.imwrite_any(out_path, sheet)
    return {"station": station, "out": str(out_path), "stats": stats}


def run_detect(station, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR,
               params=None):
    """阶段二：检测交叉点，输出标注图（筋条轴线 + 三类交叉点）。"""
    out_dir = Path(out_dir)
    enhanced, depth_mm = _load_enhanced(data_dir, station)
    res = detect.detect_rebar_intersections(depth_mm, enhanced, params)

    canvas = viz.to_bgr(enhanced)
    if res["skeleton"] is not None:
        canvas[res["skeleton"] > 0] = viz.COLOR_SKEL
    viz.draw_bars(canvas, res["h_lines"], res["v_lines"])
    vis = viz.draw_result(canvas, res)
    out_path = out_dir / f"station_{station}_detect.png"
    imgio.imwrite_any(out_path, vis)
    return {"station": station, "out": str(out_path), "stats": res["stats"],
            "split_mm": res["split_mm"]}


def run_report(station, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR,
               params=None):
    """阶段三：输出交付用的一页式结果图 + 交叉点坐标文件。"""
    out_dir = Path(out_dir)
    enhanced, depth_mm = _load_enhanced(data_dir, station)
    depth_vis = enhance.depth_visual(depth_mm)
    res = detect.detect_rebar_intersections(depth_mm, enhanced, params)

    sheet = viz.make_result_sheet(station, res["split_mm"], enhanced, depth_vis, res)
    out_path = out_dir / f"station_{station}_result.png"
    imgio.imwrite_any(out_path, sheet)

    csv_path = out_dir / f"station_{station}_points.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("point_id,x_px,y_px,z_mm\n")
        for i, (x, y, z) in enumerate(res["accepted"], 1):
            f.write(f"P{i:03d},{x},{y},{z:.2f}\n")

    json_path = out_dir / f"station_{station}_points.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"station": station,
                   "split_mm": round(res["split_mm"], 1),
                   "points": [[int(x), int(y)] for x, y, _ in res["accepted"]]},
                  f, ensure_ascii=False, indent=1)

    rej_path = out_dir / f"station_{station}_rejected.json"
    with open(rej_path, "w", encoding="utf-8") as f:
        json.dump({"lower": [[int(x), int(y)] for x, y, _ in res["rejected_lower"]],
                   "clutter": [[int(x), int(y)] for x, y, _ in res["rejected_clutter"]]},
                  f, ensure_ascii=False, indent=1)

    return {"station": station, "out": str(out_path), "csv": str(csv_path),
            "json": str(json_path), "rejected": str(rej_path),
            "stats": res["stats"], "split_mm": res["split_mm"]}


def run_all_reports(stations=None, data_dir=DEFAULT_DATA_DIR,
                    out_dir=DEFAULT_OUT_DIR, params=None):
    """批量出报告，并汇总一张 statistic 表。"""
    stations = stations or imgio.list_stations(data_dir)
    rows = []
    for n in stations:
        r = run_report(n, data_dir=data_dir, out_dir=out_dir, params=params)
        s = r["stats"]
        rows.append({
            "station": n, "split_mm": round(r["split_mm"], 1),
            "top_h_bars": s["h_bars"], "top_v_bars": s["v_bars"],
            "far_h_bars": s["far_h_bars"], "far_v_bars": s["far_v_bars"],
            "accepted": s["accepted"], "rejected_lower": s["lower"],
            "rejected_clutter": s["clutter"],
        })
        print(f"  station {n:2d}: 保留 {s['accepted']:3d}  下层 {s['lower']:3d}  "
              f"干扰 {s['clutter']:2d}  -> {Path(r['out']).name}")

    summary = Path(out_dir) / "summary.csv"
    with open(summary, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n汇总表: {summary}")
    return rows


def run_stations(stations, stage="preview", **kwargs):
    fn = {"preview": run_preview, "detect": run_detect, "report": run_report}.get(stage)
    if fn is None:
        raise NotImplementedError(f"未知阶段 {stage}")
    return [fn(n, **kwargs) for n in stations]
