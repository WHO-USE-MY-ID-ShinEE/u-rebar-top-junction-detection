"""图像读写与深度/灰度对齐。

本模块只做两件已验证的事情：
  1. 绕开 OpenCV 在 Windows 下无法读取非 ASCII 路径的问题；
  2. 把原始的横向深度图旋转成与灰度图逐像素对齐的竖构图。

对齐结论（已实测，见 docs/交接文档.md "已钉死的硬事实"）：
  深度原图 1080x1440，灰度图 1440x1080，尺寸恰好互为转置；
  深度图逆时针旋转 90 度后与灰度图逐像素对齐，无需平移或缩放。
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

DEPTH_SUFFIX = "output_depth_raw.tif"
GRAY_SUFFIX = "output_image_left.png"


def imread_any(path, flags=cv2.IMREAD_UNCHANGED):
    """读取图像，支持非 ASCII 路径。读失败返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_any(path, image):
    """写入图像，支持非 ASCII 路径。返回是否成功。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def to_gray(image):
    """把 BGR 图或灰度图统一成单通道 uint8。"""
    if image is None:
        return None
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def align_depth(depth):
    """把原始深度图旋转到与灰度图同向的构图。"""
    return cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)


def to_mm(depth):
    """深度原图单位是米，转成毫米的 float32；无效像素保持 NaN。"""
    return (depth.astype(np.float32) * 1000.0).astype(np.float32)


def station_paths(data_dir, station):
    """返回 (深度图路径, 灰度图路径)。"""
    data_dir = Path(data_dir)
    return (data_dir / f"station_{station}_{DEPTH_SUFFIX}",
            data_dir / f"station_{station}_{GRAY_SUFFIX}")


def load_pair(depth_path, gray_path=None):
    """读取一对配套的（深度图, 灰度图），返回 (gray, depth_mm, depth_raw)。

    gray      : uint8 单通道，已与深度对齐；不给灰度图时为 None
    depth_mm  : float32，单位毫米，无效像素为 NaN
    depth_raw : float32 原始米制深度（未旋转），仅用于回溯排查

    对齐规则：深度图与灰度图尺寸互为转置时逆时针旋转 90 度（本数据集的情形）；
    尺寸已相同时不旋转；都不符合则报错。
    """
    depth_raw = imread_any(depth_path)
    if depth_raw is None:
        raise FileNotFoundError(f"深度图读取失败: {depth_path}")
    depth = depth_raw
    if gray_path is not None:
        raw_gray = imread_any(gray_path)
        if raw_gray is None:
            raise FileNotFoundError(f"灰度图读取失败: {gray_path}")
        gray = to_gray(raw_gray)
        if depth_raw.shape == gray.shape:
            pass
        elif depth_raw.shape[::-1] == gray.shape:
            depth = align_depth(depth_raw)
        else:
            raise ValueError(f"深度图与灰度图尺寸无法对齐: "
                             f"depth={depth_raw.shape} gray={gray.shape}")
    else:
        gray = None
        depth = align_depth(depth_raw)      # 无灰度图可参照，按本数据集约定处理
    return gray, to_mm(depth), depth_raw


def load_station(data_dir, station):
    """读取数据集里的一个工位，返回 (gray, depth_mm, depth_raw)。"""
    depth_path, gray_path = station_paths(data_dir, station)
    return load_pair(depth_path, gray_path)


def list_stations(data_dir):
    """列出数据目录中实际存在的站点编号。"""
    data_dir = Path(data_dir)
    out = []
    for p in sorted(data_dir.glob(f"station_*_{DEPTH_SUFFIX}")):
        try:
            out.append(int(p.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def valid_mask(depth_mm):
    """有效深度掩膜：有限、大于 0。"""
    return np.isfinite(depth_mm) & (depth_mm > 0)
