"""结果可视化。

任务书要求：
  * 在可视化窗口中加载原始灰度图、深度图；
  * 把检测到的交叉点标在图上；
  * 能区分显示被过滤掉的下层交叉点、虚假干扰点；
  * 输出图像清晰美观，能直观看得到钢筋骨架和交叉点位置。

注：OpenCV 的 putText 画不了中文，这里用 PIL + 系统字体渲染中文标注，
找不到字体时自动退回 OpenCV（英文）。
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# BGR
COLOR_ACCEPTED = (0, 0, 255)      # 红：保留的顶层交叉点
COLOR_LOWER = (255, 128, 0)       # 蓝：深度阈值筛掉的下层交叉点
COLOR_CLUTTER = (0, 165, 255)     # 橙：判为虚假干扰的点
COLOR_H_BAR = (255, 0, 255)       # 品红：横筋轴线
COLOR_V_BAR = (255, 255, 0)       # 青：竖筋轴线
COLOR_SKEL = (0, 255, 0)          # 绿：骨架
COLOR_TEXT = (0, 255, 255)        # 黄：文字

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    r"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)


@lru_cache(maxsize=16)
def _font(size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def draw_text(image, text, org, size=26, color=COLOR_TEXT, stroke=2):
    """画文字，优先用中文字体；找不到字体时退回 OpenCV（仅英文可读）。"""
    font = _font(size)
    if font is None:
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0,
                    color, stroke, cv2.LINE_AA)
        return image
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]),
              stroke_width=1, stroke_fill=(0, 0, 0))
    image[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return image


def to_bgr(gray):
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray.copy()


def draw_points(canvas, points, color, radius=10, label=None, filled=True):
    """画一组点。points 元素为 (x, y) 或 (x, y, depth_mm)。"""
    for i, p in enumerate(points):
        x, y = int(round(p[0])), int(round(p[1]))
        if filled:
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius + 2, (255, 255, 255), 2, cv2.LINE_AA)
        if label and len(p) > 2 and np.isfinite(p[2]):
            cv2.putText(canvas, f"{label}{i + 1}:{p[2]:.0f}", (x + radius + 6, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2, cv2.LINE_AA)
    return canvas


def draw_bars(canvas, h_lines, v_lines, thickness=3):
    """画筋条轴线：品红=横筋，青=竖筋。"""
    for lines, color in ((h_lines, COLOR_H_BAR), (v_lines, COLOR_V_BAR)):
        for l in lines:
            p0, p1 = l["extent"]
            cv2.line(canvas, tuple(np.round(p0).astype(int)),
                     tuple(np.round(p1).astype(int)), color, thickness, cv2.LINE_AA)
    return canvas


def draw_legend(canvas, entries, org=(18, 60), size=24, pad=10):
    """在画布上画图例。entries 为 [(BGR颜色, 文字)]。"""
    font = _font(size)
    line_h = size + pad
    box_h = line_h * len(entries) + pad
    width = 0
    for _, text in entries:
        if font is not None:
            width = max(width, int(font.getlength(text)))
        else:
            width = max(width, int(len(text) * size * 0.6))
    x0, y0 = org
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 8), (x0 + width + 46, y0 + box_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, (color, text) in enumerate(entries):
        cy = y0 + line_h * i + size // 2
        cv2.circle(canvas, (x0 + 10, cy), size // 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x0 + 10, cy), size // 3 + 1, (255, 255, 255), 1, cv2.LINE_AA)
        draw_text(canvas, text, (x0 + 30, y0 + line_h * i), size, (255, 255, 255))
    return canvas


def result_panel(gray_enh, result, title=None, with_labels=True, scale=1.0):
    """主结果面板：增强灰度 + 筋轴线 + 三类交叉点 + 图例。"""
    canvas = to_bgr(gray_enh)
    if result.get("skeleton") is not None:
        canvas[result["skeleton"] > 0] = (60, 90, 60)
    draw_bars(canvas, result["h_lines"], result["v_lines"])
    draw_points(canvas, result["rejected_clutter"], COLOR_CLUTTER, radius=8)
    draw_points(canvas, result["rejected_lower"], COLOR_LOWER, radius=8)
    draw_points(canvas, result["accepted"], COLOR_ACCEPTED, radius=10,
                label="P" if with_labels else None)
    if title:
        draw_text(canvas, title, (18, 14), 30)
    draw_legend(canvas, [
        (COLOR_ACCEPTED, f"顶层交叉点 {len(result['accepted'])}"),
        (COLOR_LOWER, f"筛掉的下层点 {len(result['rejected_lower'])}"),
        (COLOR_CLUTTER, f"虚假干扰点 {len(result['rejected_clutter'])}"),
        (COLOR_H_BAR, f"横筋轴线 {len(result['h_lines'])}"),
        (COLOR_V_BAR, f"竖筋轴线 {len(result['v_lines'])}"),
    ])
    return canvas


def layer_panel(depth_vis, top_mask, far_mask, title=None):
    """分层面板：深度伪彩 + 顶层/下层掩膜着色。"""
    canvas = to_bgr(depth_vis)
    canvas[top_mask] = (canvas[top_mask] * 0.35
                        + np.array([0, 0, 255]) * 0.65).astype(np.uint8)
    canvas[far_mask] = (canvas[far_mask] * 0.35
                        + np.array([255, 128, 0]) * 0.65).astype(np.uint8)
    if title:
        draw_text(canvas, title, (18, 14), 30)
    return canvas


def mask_panel(gray_enh, top_mask, result, title=None):
    """中间结果面板：顶层掩膜 + 骨架 + 筋轴线。"""
    canvas = to_bgr(gray_enh)
    canvas[top_mask] = (canvas[top_mask] * 0.35
                        + np.array([0, 90, 0]) * 0.65).astype(np.uint8)
    if result.get("skeleton") is not None:
        canvas[result["skeleton"] > 0] = COLOR_SKEL
    draw_bars(canvas, result["h_lines"], result["v_lines"])
    if title:
        draw_text(canvas, title, (18, 14), 30)
    return canvas


def info_panel(lines, size, title=None):
    """统计信息面板（纯文字）。"""
    w, h = size
    canvas = np.full((h, w, 3), 245, np.uint8)
    draw_text(canvas, title or "统计", (24, 20), 34, (30, 30, 30))
    y = 90
    for text, color, sz in lines:
        draw_text(canvas, text, (30, y), sz, color)
        y += sz + 16
    return canvas


def side_by_side_gray_depth(gray, depth_vis, gap=10):
    a, b = to_bgr(gray), to_bgr(depth_vis)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return np.hstack([a, np.full((a.shape[0], gap, 3), 255, np.uint8), b])


def fit_width(image, width=900):
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def make_result_sheet(station, split_mm, gray_enh, depth_vis, result,
                      panel_width=900, gap=12):
    """把四个面板拼成一页式结果图，作为交付用的测试输出。"""
    top_mask, far_mask = result["top_mask"], result["far_mask"]
    panels = [
        result_panel(gray_enh, result, f"{station} 号工位 · 顶层交叉点检测结果"),
        layer_panel(depth_vis, top_mask, far_mask,
                    f"深度分层（分界 {split_mm:.0f} mm）"),
        mask_panel(gray_enh, top_mask, result, "顶层掩膜 + 骨架 + 筋轴线"),
        None,
    ]
    resized = [fit_width(p, panel_width) for p in panels[:3]]
    ph = max(p.shape[0] for p in resized)
    stats = result["stats"]
    info = info_panel([
        (f"工位：{station}", (30, 30, 30), 26),
        (f"深度分界：{split_mm:.1f} mm", (30, 30, 30), 26),
        (f"顶层横筋 / 竖筋：{stats['h_bars']} / {stats['v_bars']} 条",
         (140, 30, 30), 26),
        (f"顶层交叉点（保留）：{stats['accepted']} 个", (0, 0, 200), 30),
        ("", (0, 0, 0), 10),
        (f"筛掉的下层交叉点：{stats['lower']} 个", (170, 80, 0), 26),
        (f"筛掉的虚假干扰点：{stats['clutter']} 个", (0, 100, 200), 26),
        ("", (0, 0, 0), 10),
        (f"下层横筋 / 竖筋：{stats['far_h_bars']} / {stats['far_v_bars']} 条",
         (90, 90, 90), 24),
        ("", (0, 0, 0), 10),
        ("红=保留  蓝=下层  橙=干扰", (60, 60, 60), 22),
        ("品红=横筋轴线  青=竖筋轴线", (60, 60, 60), 22),
        ("绿=顶层骨架", (60, 60, 60), 22),
    ], (panel_width, ph))
    resized.append(info)

    sheet = np.full((ph * 2 + gap, panel_width * 2 + gap, 3), 255, np.uint8)
    for i, p in enumerate(resized):
        r, c = divmod(i, 2)
        y = r * (ph + gap)
        x = c * (panel_width + gap)
        sheet[y:y + p.shape[0], x:x + p.shape[1]] = p
    return sheet
