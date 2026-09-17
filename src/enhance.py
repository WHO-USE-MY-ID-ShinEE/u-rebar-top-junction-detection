"""灰度图增强。

原始灰度图极暗（均值 16.7，87% 像素 < 20），直接做边缘检测几乎提不出钢筋。
实测 CLAHE 效果显著：螺纹钢的横肋纹理、背后网片、圆形垫块都变得清晰可辨，
因此光照校正是必需步骤而非可选步骤。
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_CLIP_LIMIT = 3.0
DEFAULT_TILE_GRID = (8, 8)


def clahe(gray, clip_limit=DEFAULT_CLIP_LIMIT, tile_grid=DEFAULT_TILE_GRID):
    """对比度受限自适应直方图均衡。输入输出均为 uint8 单通道。"""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(gray)


def to_uint8(image):
    """把任意量程的图线性映射到 0-255，用于显示。"""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def texture_energy(gray, ksize=9):
    """局部方差作为纹理能量，用于突出螺纹钢的横肋与边缘。

    注意：本指标对钢筋网格有周期性混叠，不能用于估计深度/灰度之间的平移量。
    """
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (ksize, ksize))
    mean_sq = cv2.boxFilter(f * f, -1, (ksize, ksize))
    return np.clip(mean_sq - mean * mean, 0.0, None)


def enhance(gray, clip_limit=DEFAULT_CLIP_LIMIT,
            tile_grid=DEFAULT_TILE_GRID):
    """返回 (增强灰度图, 纹理能量图)。clip_limit 可由界面调节。"""
    enhanced = clahe(gray, clip_limit=clip_limit, tile_grid=tile_grid)
    return enhanced, texture_energy(enhanced)


def depth_visual(depth_mm, lo=None, hi=None):
    """把深度图映射成伪彩图用于显示，无效像素置黑。"""
    from .imgio import valid_mask

    vis = np.zeros(depth_mm.shape, np.uint8)
    m = valid_mask(depth_mm)
    if not m.any():
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vals = depth_mm[m]
    if lo is None:
        lo = float(np.percentile(vals, 2))
    if hi is None:
        hi = float(np.percentile(vals, 98))
    span = max(hi - lo, 1e-6)
    vis[m] = np.clip((vals - lo) / span * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)
