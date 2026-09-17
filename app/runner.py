"""应用的计算入口：加载 -> 检测 -> 审计 -> 导出。

界面（app/gui.py）与无界面自检（app/selftest.py）都走这里，保证两条路径结果一致。
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from src import audit, detect, enhance, imgio, pipeline, viz

from . import render

# 界面上可调、且确实影响结果的参数（键名与 detect.DEFAULTS 保持一致）
TUNABLE = (
    ("split_mm", "深度分界 (mm)", "空 = 按深度空段自动定界"),
    ("hough_thresh", "Hough 阈值", "越大越严格，筋条检出更少"),
    ("hough_min_len", "Hough 最小段长 (px)", "太短会把杂物碎块当筋条"),
    ("min_coverage", "筋条覆盖度下限", "支撑像素数 / 筋条跨度"),
    ("support_min_ratio", "交叉点支撑比下限", "两个方向都要达到，滤掉断线相交"),
    ("blob_fill_limit", "实心块填充率上限", "四角填充率超限判为实心杂物"),
)
TUNABLE_EXTRA = (
    ("clahe_clip", "CLAHE 对比度", "灰度增强的对比度上限，空 = 默认 3.0"),
)
# 这些参数必须是整数（其余按浮点解析）
INT_KEYS = ("hough_thresh", "hough_min_len")
DEFAULTS_TEXT = {"split_mm": ""}       # 空串表示"自动/默认"


def name_for_paths(depth_path):
    """由文件路径生成输出用的短名字。"""
    stem = Path(depth_path).stem
    return stem.replace("_output_depth_raw", "") or Path(depth_path).name


def default_text(key):
    """界面输入框的默认文本。"""
    if key in DEFAULTS_TEXT:
        return DEFAULTS_TEXT[key]
    value = detect.DEFAULTS.get(key)
    if value is None:
        value = enhance.DEFAULT_CLIP_LIMIT
    return str(value)


def parse_params(texts):
    """把界面上的参数字符串转成参数字典；空串表示用默认值。"""
    params = {}
    for key, raw in texts.items():
        raw = str(raw).strip()
        if not raw:
            continue
        try:
            params[key] = int(raw) if key in INT_KEYS else float(raw)
        except ValueError as exc:
            raise ValueError(f"参数「{key}」不是数字：{raw}") from exc
    return params


def load(data_dir=None, station=None, depth_path=None, gray_path=None,
         clahe_clip=None):
    """只加载并增强图像，不做检测（界面选中样本后可立刻预览）。

    两种输入方式：
      * 数据集工位：给 data_dir + station
      * 任意图像对：给 depth_path（可选 gray_path）
    """
    if depth_path is not None:
        gray, depth_mm, _ = imgio.load_pair(depth_path, gray_path)
        name = name_for_paths(depth_path)
        source = {"kind": "pair", "depth": str(depth_path),
                  "gray": None if gray_path is None else str(gray_path)}
    else:
        gray, depth_mm, _ = imgio.load_station(data_dir, station)
        name = f"station_{station}"
        source = {"kind": "station", "data_dir": str(data_dir), "station": station}

    enhanced = None
    if gray is not None:
        kwargs = {} if clahe_clip is None else {"clip_limit": clahe_clip}
        enhanced, _ = enhance.enhance(gray, **kwargs)
    return {
        "name": name, "source": source, "gray": gray, "enhanced": enhanced,
        "depth_mm": depth_mm, "depth_vis": enhance.depth_visual(depth_mm),
        "res": None, "audit": None, "stats": None, "params": {}, "elapsed": 0.0,
    }


def detect_run(state, params=None):
    """在已加载的图像上跑检测与完备性审计，结果写回 state 并返回。"""
    t0 = time.time()
    res = detect.detect_rebar_intersections(state["depth_mm"], state["enhanced"],
                                            params)
    state["res"] = res
    state["audit"] = audit.audit_result(state["depth_mm"], res, params)
    state["stats"] = res["stats"]
    state["params"] = dict(params or {})
    state["elapsed"] = time.time() - t0
    return state


def analyze(data_dir=None, station=None, depth_path=None, gray_path=None,
            params=None, clahe_clip=None):
    """一步跑完加载 + 检测，返回界面/导出需要的全部内容。"""
    state = load(data_dir=data_dir, station=station, depth_path=depth_path,
                 gray_path=gray_path, clahe_clip=clahe_clip)
    return detect_run(state, params)


def audit_row(state):
    """把一个分析结果压成批量统计表里的一行。"""
    s, a = state["stats"], state["audit"]
    return {
        "name": state["name"],
        "split_mm": round(state["res"]["split_mm"], 1),
        "h_bars": s["h_bars"], "v_bars": s["v_bars"],
        "far_h_bars": s["far_h_bars"], "far_v_bars": s["far_v_bars"],
        "accepted": s["accepted"], "lower": s["lower"], "clutter": s["clutter"],
        "expected": a["expected"], "matched": a["matched"],
        "misses": a["misses"], "extra": len(a["extra"]),
        "completeness": round(a["completeness"] * 100, 2),
        "elapsed": round(state["elapsed"], 2),
    }


def batch(stations, data_dir, params=None, clahe_clip=None,
          on_progress=None, should_stop=None):
    """批量分析数据集工位，返回 (统计行列表, 分析结果列表)。"""
    rows, states = [], []
    for i, n in enumerate(stations, 1):
        if should_stop is not None and should_stop():
            break
        if on_progress is not None:
            on_progress(f"正在检测 station_{n} …（{i}/{len(stations)}）",
                        i - 1, len(stations))
        st = analyze(data_dir=data_dir, station=n, params=params,
                     clahe_clip=clahe_clip)
        states.append(st)
        rows.append(audit_row(st))
    if on_progress is not None:
        on_progress("批量检测完成", len(stations), len(stations))
    return rows, states


def export(state, out_dir):
    """导出一个分析结果：一页式结果图 + 坐标 csv/json + 被剔除点 json。

    工位数据集用站号当标题（与命令行流水线的交付产物逐字节一致），
    打开任意图像对时没有站号，就用文件名。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = pipeline.export_points(out_dir, state["name"], state["res"])
    source = state["source"]
    title = source["station"] if source.get("kind") == "station" else state["name"]
    sheet = viz.make_result_sheet(title, state["res"]["split_mm"],
                                  render.base_gray(state), state["depth_vis"],
                                  state["res"])
    png = out_dir / f"{state['name']}_result.png"
    imgio.imwrite_any(png, sheet)
    files["result_png"] = str(png)
    return files


SUMMARY_FIELDS = ["name", "split_mm", "h_bars", "v_bars", "far_h_bars",
                  "far_v_bars", "accepted", "lower", "clutter", "expected",
                  "matched", "misses", "extra", "completeness", "elapsed"]


def write_summary(rows, out_dir, filename="app_summary.csv"):
    """把批量统计行写成汇总表（utf-8-sig 便于 Excel 直接打开）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path
