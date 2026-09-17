"""结果可视化。

任务书要求：在原灰度图上标记交叉点，且能区分显示被过滤掉的下层交叉点
与虚假干扰点，便于算法调试与效果分析。
"""
from __future__ import annotations

import cv2
import numpy as np

# BGR
COLOR_ACCEPTED = (0, 0, 255)    # 红：保留的顶层交叉点
COLOR_LOWER = (255, 128, 0)     # 蓝：被判为下层的干扰点
COLOR_CLUTTER = (0, 165, 255)   # 橙：被判为非钢筋杂物的干扰点
COLOR_LABEL = (0, 255, 255)     # 黄：文字


def to_bgr(gray):
    """单通道转三通道 BGR。"""
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray.copy()


def draw_points(canvas, points, color, radius=10, label=None, filled=True):
    """在画布上画一组点。points 元素为 (x, y) 或 (x, y, depth_mm)。"""
    for i, p in enumerate(points):
        x, y = int(round(p[0])), int(round(p[1]))
        if filled:
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius + 2, (255, 255, 255), 2, cv2.LINE_AA)
        if label and len(p) > 2 and np.isfinite(p[2]):
            cv2.putText(canvas, f"{label}{i + 1}:{p[2]:.0f}", (x + radius + 6, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 2, cv2.LINE_AA)
    return canvas


def draw_result(gray, result, title=None):
    """按任务书要求叠加显示：保留点 + 两类被剔除点。

    result 为 detect.detect_intersections 的返回字典。
    """
    canvas = to_bgr(gray)
    draw_points(canvas, result.get("rejected_clutter", []), COLOR_CLUTTER,
                radius=8, label="C")
    draw_points(canvas, result.get("rejected_lower", []), COLOR_LOWER,
                radius=8, label="L")
    draw_points(canvas, result.get("accepted", []), COLOR_ACCEPTED,
                radius=10, label="P")

    if title is None:
        title = (f"accepted: {len(result.get('accepted', []))}  "
                 f"lower: {len(result.get('rejected_lower', []))}  "
                 f"clutter: {len(result.get('rejected_clutter', []))}")
    cv2.putText(canvas, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                COLOR_LABEL, 3, cv2.LINE_AA)
    return canvas


def side_by_side_gray_depth(gray, depth_vis, gap=10):
    """灰度图与深度伪彩图并排，用于对照检查。"""
    a, b = to_bgr(gray), to_bgr(depth_vis)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return np.hstack([a, np.full((a.shape[0], gap, 3), 255, np.uint8), b])


def fit_width(image, width=900):
    """等比缩放到指定宽度，便于查看。"""
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
