"""顶层/下层钢筋的深度分离。

物理依据：深度图只对钢筋表面有回波，无效像素（NaN）即背景。
实测每个站点的深度直方图都是分簇的，簇与簇之间有一段空白，因此可以按
"第一段足够宽的空白"把离相机最近的那一层（顶层，待捆扎层）切出来。

注意相机是斜视的：同一层内深度跨度可达 110mm（station_1 近层中位数 753.6mm，
尾部到 790mm 以上），所以阈值必须逐站点自适应，不能写死全局常数。
"""
from __future__ import annotations

import numpy as np

from .imgio import valid_mask

DEFAULT_SPLIT_MM = 840.0   # 兜底默认值，仅在自动定界失败时使用
GAP_BIN_MM = 2.0           # 深度直方图分箱宽度
MIN_GAP_MM = 20.0          # 空段宽度下限，超过才认为两层可由深度分开（钢筋直径 16mm）
MIN_FAR_RATIO = 0.05       # 空段远侧至少要占这么多有效像素，否则视为分布尾部的空白


def _histogram(depth_mm, bin_mm=GAP_BIN_MM):
    """在稳健量程内统计深度直方图，返回 (计数, 分箱左边界)。"""
    m = valid_mask(depth_mm)
    if not m.any():
        return None, None
    v = depth_mm[m]
    lo = float(np.percentile(v, 0.1))
    hi = float(np.percentile(v, 99.9))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None
    edges = np.arange(lo, hi + bin_mm, bin_mm)
    hist, _ = np.histogram(v, bins=edges)
    return hist, edges


def top_layer_gap(depth_mm, bin_mm=GAP_BIN_MM, min_gap_mm=MIN_GAP_MM,
                  min_far_ratio=MIN_FAR_RATIO):
    """找出把顶层和其余层分开的那段深度空白。

    候选必须是"被两侧占据区间夹住"的空白，且空白之后还有足够多的像素
    （排除分布尾部那种"什么都没有"的空白）。
    返回 (空段宽度_mm, 空段起点_mm, 分层阈值_mm)，无合适空段时阈值返回 None。
    """
    hist, edges = _histogram(depth_mm, bin_mm)
    if hist is None:
        return 0.0, float("nan"), None

    occupied = hist > 0
    total = int(hist.sum())
    min_bins = max(int(round(min_gap_mm / bin_mm)), 1)

    runs = []                      # (起始bin, 长度bin)
    start = None
    for i, occ in enumerate(occupied):
        if not occ:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - start))
            start = None

    for s, length in runs:
        if length < min_bins:
            continue
        if s == 0 or s + length >= len(occupied):
            continue               # 贴着量程边界的空白不算层间空白
        far_px = int(hist[s + length:].sum())
        if far_px < min_far_ratio * total:
            continue               # 空白之后几乎是空的，属于分布尾部
        width = length * bin_mm
        return width, float(edges[s]), float(edges[s] + width / 2.0)

    return 0.0, float("nan"), None


def auto_split_mm(depth_mm, **kwargs):
    """自动分层阈值；无合适空段时返回 None（需改用顶层平面拟合）。"""
    return top_layer_gap(depth_mm, **kwargs)[2]


def split_layers(depth_mm, split_mm=DEFAULT_SPLIT_MM):
    """按深度阈值分成近层/远层，返回 (near_mask, far_mask, valid_mask)。"""
    m = valid_mask(depth_mm)
    return m & (depth_mm < split_mm), m & (depth_mm >= split_mm), m


def _describe(depth_mm, mask):
    if not mask.any():
        return {"px": 0, "median_mm": float("nan"),
                "p2_mm": float("nan"), "p98_mm": float("nan")}
    v = depth_mm[mask]
    return {"px": int(mask.sum()), "median_mm": float(np.median(v)),
            "p2_mm": float(np.percentile(v, 2)),
            "p98_mm": float(np.percentile(v, 98))}


def layer_stats(depth_mm, split_mm=None):
    """分层统计。split_mm 为 None 时自动定界。"""
    m = valid_mask(depth_mm)
    gap_width, gap_start, auto = top_layer_gap(depth_mm)
    used = split_mm if split_mm is not None else (auto if auto is not None
                                                 else DEFAULT_SPLIT_MM)
    near, far, _ = split_layers(depth_mm, used)
    near_s, far_s = _describe(depth_mm, near), _describe(depth_mm, far)
    total = int(m.sum())
    return {"valid_px": total, "near": near_s, "far": far_s,
            "split_mm": float(used), "auto_split_mm": auto,
            "gap_width_mm": gap_width, "gap_start_mm": gap_start,
            "near_ratio": near_s["px"] / total if total else 0.0,
            "separable": bool(auto is not None and near_s["px"] > far_s["px"])}


def top_layer_mask(depth_mm, split_mm=None):
    """顶层（待捆扎层）掩膜，即离相机最近的那一层钢筋。

    TODO(路线图第 2 步): 改用拟合顶层深度平面 f(x, y) 后按
    depth - f(x, y) < tol 判定，彻底消除斜视带来的层内深度跨度。
    """
    if split_mm is None:
        split_mm = auto_split_mm(depth_mm) or DEFAULT_SPLIT_MM
    near, _, _ = split_layers(depth_mm, split_mm)
    return near
