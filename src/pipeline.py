"""流程编排：把各模块串成可运行的单站/批量流程。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import detect, enhance, imgio, layers, viz

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "任务书数据+代码"
DEFAULT_OUT_DIR = ROOT / "outputs"


def _stack(panels, pad_color=(255, 255, 255)):
    """把若干图纵向拼接，右侧用白边补齐。"""
    width = max(p.shape[1] for p in panels)
    rows = []
    for p in panels:
        rows.append(cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1],
                                       cv2.BORDER_CONSTANT, value=pad_color))
        rows.append(np.full((10, width, 3), pad_color, np.uint8))
    return np.vstack(rows[:-1])


def run_preview(station, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR,
                split_mm=None):
    """阶段一：读数据 -> 灰度增强 -> 分层 -> 输出对照图。split_mm 为 None 时自动定界。"""
    out_dir = Path(out_dir)
    gray, depth_mm, _ = imgio.load_station(data_dir, station)
    enhanced, _ = enhance.enhance(gray)
    depth_vis = enhance.depth_visual(depth_mm)

    stats = layers.layer_stats(depth_mm, split_mm)
    near, far, _ = layers.split_layers(depth_mm, stats["split_mm"])

    overlay = viz.to_bgr(enhanced)
    overlay[near] = (overlay[near] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    overlay[far] = (overlay[far] * 0.4 + np.array([255, 128, 0]) * 0.6).astype(np.uint8)
    cv2.putText(overlay, f"red = top layer (<{stats['split_mm']:.0f}mm)   blue = lower layer",
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
    gray, depth_mm, _ = imgio.load_station(data_dir, station)
    enhanced, _ = enhance.enhance(gray)
    res = detect.detect_rebar_intersections(depth_mm, enhanced, params)

    canvas = viz.to_bgr(enhanced)
    skel = res["skeleton"]
    if skel is not None:
        canvas[skel > 0] = (0, 255, 0)
    for l in res["h_lines"]:
        p0, p1 = l["extent"]
        cv2.line(canvas, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                 (255, 0, 255), 3, cv2.LINE_AA)
    for l in res["v_lines"]:
        p0, p1 = l["extent"]
        cv2.line(canvas, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                 (255, 255, 0), 3, cv2.LINE_AA)

    vis = viz.draw_result(canvas, res)
    out_path = out_dir / f"station_{station}_detect.png"
    imgio.imwrite_any(out_path, vis)
    csv_path, json_path, rej_path = export_points(station, res, out_dir)
    return {"station": station, "out": str(out_path), "csv": str(csv_path),
            "json": str(json_path), "rejected": str(rej_path),
            "stats": res["stats"], "split_mm": res["split_mm"]}


def export_points(station, result, out_dir=DEFAULT_OUT_DIR):
    """按任务书要求输出交叉点坐标：csv 保存点位集合，json 每行一个 [x, y]。"""
    out_dir = Path(out_dir)
    points = result["accepted"]

    csv_path = out_dir / f"station_{station}_points.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("point_id,x_px,y_px,z_mm\n")
        for i, (x, y, z) in enumerate(points, 1):
            f.write(f"A{i:03d},{x},{y},{z:.2f}\n")

    json_path = out_dir / f"station_{station}_points.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"station": station,
                   "points": [[int(x), int(y)] for x, y, _ in points]},
                  f, ensure_ascii=False, indent=1)

    # 被剔除的两类点也留档，便于报告里做对比分析
    rej_path = out_dir / f"station_{station}_rejected.json"
    with open(rej_path, "w", encoding="utf-8") as f:
        json.dump({"lower": [[int(x), int(y)] for x, y, _ in result["rejected_lower"]],
                   "clutter": [[int(x), int(y)] for x, y, _ in result["rejected_clutter"]]},
                  f, ensure_ascii=False, indent=1)
    return csv_path, json_path, rej_path


def run_stations(stations, stage="preview", **kwargs):
    fn = {"preview": run_preview, "detect": run_detect}.get(stage)
    if fn is None:
        raise NotImplementedError(f"阶段 {stage} 尚未实现，见 docs/交接文档.md")
    return [fn(n, **kwargs) for n in stations]
