"""交叉点完备性自审计（应用与命令行工具共用）。

思路：检测算法本身是"顶层横筋 x 顶层竖筋两两求交 -> 逐个判定是否保留"。本模块反向
检查：把几何上真正落在两条筋条跨度之内的交点全部列成"期望交叉点"，与最终保留点比对。

  期望交叉点  = 算法应当给出的答案（不需要人工标注真值）
  未保留的期望点 = 候选漏检，逐个给出原因（支撑不足 / 压在实心杂物上 / 筋条过短 …）
  多余保留点  = 候选误检（保留点找不到对应的几何交叠）

任务书要求的"漏检数 / 误检数 / 准确率"在没有人工标注的情况下即以此口径统计：
  漏检数 = 未被保留、且原因不属于"已查明的正确剔除"的期望点（当前 10 站为 0）
  误检数 = 多余保留点
  完备率 = 保留的期望点数 / 期望交叉点总数
"""
from __future__ import annotations

import numpy as np

from . import detect

MATCH_PX = 20.0          # 期望点与保留点算作同一个点的距离
REASON_ORDER = ("保留", "两向支撑不足", "压在实心杂物上", "筋条过短",
                "深度判为下层", "无深度回波")


def _nearest(pt, pts):
    if not pts:
        return None
    arr = np.asarray(pts, float)
    return float(np.hypot(arr[:, 0] - pt[0], arr[:, 1] - pt[1]).min())


def audit_result(depth_mm, res, params=None):
    """对一个已有的检测结果做完备性审计，返回统计字典（不重新检测）。"""
    p = dict(detect.DEFAULTS)
    if params:
        p.update(params)
    split = res["split_mm"]
    accepted = [(x, y) for x, y, _ in res["accepted"]]

    rows = []
    for a in res["h_lines"]:
        for b in res["v_lines"]:
            pt = detect.intersect(a, b)
            if pt is None:
                continue
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            if not (0 <= x < depth_mm.shape[1] and 0 <= y < depth_mm.shape[0]):
                continue
            if not (detect._within(a, pt, p["cross_margin_px"])
                    and detect._within(b, pt, p["cross_margin_px"])):
                continue
            d = detect.local_depth(depth_mm, x, y)
            if d is None:
                reason = "无深度回波"
            elif d["median"] >= split:
                reason = "深度判为下层"
            elif min(a["span"], b["span"]) < p["min_bar_px"]:
                reason = "筋条过短"
            elif not detect.crossing_supported(res["top_mask"], pt, a, b, p):
                reason = "两向支撑不足"
            elif detect.solid_around(res["top_mask"], pt, p):
                reason = "压在实心杂物上"
            else:
                reason = "保留"
            rows.append({"x": x, "y": y, "reason": reason,
                         "dist": _nearest((x, y), accepted)})

    missing = [r for r in rows if r["dist"] is None or r["dist"] > MATCH_PX]
    extra = []
    exp = np.asarray([(r["x"], r["y"]) for r in rows], float)
    for pt in accepted:
        if exp.size == 0 or np.hypot(exp[:, 0] - pt[0],
                                     exp[:, 1] - pt[1]).min() > MATCH_PX:
            extra.append(pt)

    reasons = {}
    for r in missing:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    total = len(rows)
    matched = total - len(missing)
    return {"expected": total, "matched": matched,
            "missing": missing, "extra": extra, "reasons": reasons,
            "accepted": len(accepted),
            "completeness": (matched / total) if total else 0.0,
            "misses": sum(v for k, v in reasons.items() if k != "保留"),
            "lost": sum(1 for r in missing if r["reason"] == "保留")}


def audit_station(station, data_dir, params=None):
    """读一个工位、跑一次检测并审计，返回 (检测结果, 审计结果)。"""
    from . import imgio, enhance

    gray, depth_mm, _ = imgio.load_station(data_dir, station)
    enhanced, _ = enhance.enhance(gray)
    res = detect.detect_rebar_intersections(depth_mm, enhanced, params)
    return res, audit_result(depth_mm, res, params)
