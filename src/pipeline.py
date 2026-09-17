"""流程编排：把各模块串成可运行的单站/批量流程。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import enhance, imgio, layers, viz

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
    """阶段一：读数据 -> 灰度增强 -> 分层 -> 输出对照图。

    当前已实现并可验证的最小闭环，用于确认数据读取、对齐、分层的正确性。
    split_mm 为 None 时按深度空段自动定界。返回该站点的路径与分层统计。
    """
    out_dir = Path(out_dir)
    gray, depth_mm, _ = imgio.load_station(data_dir, station)
    enhanced, _ = enhance.enhance(gray)
    depth_vis = enhance.depth_visual(depth_mm)

    stats = layers.layer_stats(depth_mm, split_mm)
    near, far, _ = layers.split_layers(depth_mm, stats["split_mm"])

    overlay = viz.to_bgr(enhanced)
    overlay[near] = (overlay[near] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    overlay[far] = (overlay[far] * 0.4 + np.array([255, 128, 0]) * 0.6).astype(np.uint8)
    cv2.putText(overlay, f"red = top layer (<{stats['split_mm']:.0f}mm)   "
                         f"blue = lower layer", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

    sheet = _stack([viz.fit_width(p) for p in
                    (viz.side_by_side_gray_depth(enhanced, depth_vis), overlay)])
    out_path = out_dir / f"station_{station}_preview.png"
    imgio.imwrite_any(out_path, sheet)
    return {"station": station, "out": str(out_path), "stats": stats}


def run_stations(stations, stage="preview", **kwargs):
    """批量跑多个站点。stage="preview" 为当前唯一已实现的阶段。"""
    if stage != "preview":
        raise NotImplementedError(f"阶段 {stage} 尚未实现，见 docs/交接文档.md")
    return [run_preview(n, **kwargs) for n in stations]
