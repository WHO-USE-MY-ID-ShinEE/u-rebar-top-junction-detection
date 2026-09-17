"""界面要显示的几种视图（纯计算，不依赖界面库，便于无界面自测）。"""
from __future__ import annotations

import cv2
import numpy as np

from src import viz

# 视图名称即界面上的选择项，顺序也一致
VIEWS = ("原始灰度图", "增强灰度图", "深度伪彩图", "检测结果")

# 界面上的显示开关。被剔除的两类点分色显示，且可以分别关掉（任务书要求能区分显示）
DEFAULT_OPTIONS = {
    "bars": True,        # 画筋条轴线
    "skeleton": False,   # 画顶层骨架（诊断用）
    "lower": True,       # 画被剔除的下层交叉点
    "clutter": True,     # 画被判为虚假干扰的点
    "labels": False,     # 给保留点编号
}


def _no_gray(depth_vis):
    """没有灰度图时，用压暗的深度伪彩当底图（保证画面上能看清结构）。"""
    return (viz.to_bgr(depth_vis).astype(np.float32) * 0.55).astype(np.uint8)


def base_image(state):
    """检测结果图的底图：有灰度图就用增强灰度，没有就用深度伪彩。"""
    if state["enhanced"] is not None:
        return viz.to_bgr(state["enhanced"])
    return _no_gray(state["depth_vis"])


def base_gray(state):
    """一页式结果图（viz.make_result_sheet）要求的单通道底图。"""
    if state["enhanced"] is not None:
        return state["enhanced"]
    return cv2.cvtColor(_no_gray(state["depth_vis"]), cv2.COLOR_BGR2GRAY)


def view_image(view, state, options=None):
    """按视图名生成 BGR 图像。state 是 app.runner.load / analyze 的返回字典。"""
    opt = dict(DEFAULT_OPTIONS)
    if options:
        opt.update(options)

    if view == VIEWS[0] and state["gray"] is not None:
        return viz.to_bgr(state["gray"])
    if view == VIEWS[1] and state["enhanced"] is not None:
        return viz.to_bgr(state["enhanced"])
    if view == VIEWS[2]:
        return viz.to_bgr(state["depth_vis"])

    # 原始/增强视图在没有对应灰度图时退回深度伪彩底图；检测结果图同理
    res = state["res"]
    if view != VIEWS[3] or res is None:
        return base_image(state)

    canvas = base_image(state)
    if opt["skeleton"] and res.get("skeleton") is not None:
        canvas[res["skeleton"] > 0] = viz.COLOR_SKEL
    if opt["bars"]:
        viz.draw_bars(canvas, res["h_lines"], res["v_lines"])
    if opt["clutter"]:
        viz.draw_points(canvas, res["rejected_clutter"], viz.COLOR_CLUTTER, radius=10)
    if opt["lower"]:
        viz.draw_points(canvas, res["rejected_lower"], viz.COLOR_LOWER, radius=10)
    viz.draw_points(canvas, res["accepted"], viz.COLOR_ACCEPTED, radius=10,
                    label="P" if opt["labels"] else None)
    viz.draw_legend(canvas, [
        (viz.COLOR_ACCEPTED, f"顶层交叉点 {len(res['accepted'])}"),
        (viz.COLOR_LOWER, f"筛掉的下层点 {len(res['rejected_lower'])}"),
        (viz.COLOR_CLUTTER, f"虚假干扰点 {len(res['rejected_clutter'])}"),
        (viz.COLOR_H_BAR, f"横筋轴线 {len(res['h_lines'])}"),
        (viz.COLOR_V_BAR, f"竖筋轴线 {len(res['v_lines'])}"),
    ])
    return canvas
