"""顶层钢筋交叉点检测。

流水线：
  1. 深度有效像素 = 所有钢筋；按深度分成顶层掩膜 / 下层掩膜（见 layers.py）；
  2. 掩膜 -> 闭运算补缺口 -> Zhang-Suen 骨架化 -> HoughLinesP 取线段；
  3. 线段按"走向族"参数化后聚类成筋条，再对筋条附近的骨架像素做 PCA 精修；
  4. 合并近重复筋条；
  5. 横筋线与竖筋线两两求交，三分类：
       保留       —— 两条筋都够长（像筋条）
       下层       —— 在下层掩膜上检出的交叉点（被深度阈值筛掉的那批）
       非钢筋杂物 —— 参与相交的筋条太短，形状不像长条形钢筋

参数化要点（踩过的坑）：
  * 不要用统一的法式参数 (theta, rho)。theta 归一到 (-90°, 90°] 后，接近竖直的筋
    会因数值抖动在 +89° / -89° 之间跳变，导致同一条筋被拆成两簇、竖筋重复检出。
    改成按族参数化：横筋拟合 y = a*x + b，竖筋拟合 x = a*y + b，歧义消失。
  * rho 的原点若取图像左上角，线段转角的微小抖动会被放大成几十像素的偏移，
    同理会把一条筋拆散。族参数化后 b 直接就是截距，没有这个问题。
"""
from __future__ import annotations

import cv2
import numpy as np

from .layers import auto_split_mm, split_layers

DEFAULTS = dict(
    close_ksize=3,
    skel_max_iter=18,
    hough_thresh=60,
    hough_min_len=150,
    hough_max_gap=30,
    cluster_slope_tol=0.05,    # 约 2.9 度
    cluster_intercept_px=20.0,
    refit_halfwidth=14,
    dedup_slope_tol=0.03,
    dedup_intercept_px=24.0,
    min_support_px=120,
    min_coverage=0.5,       # 支撑像素数/跨度 的下限，滤掉勉强连起来的弱线
    far_min_coverage=0.2,   # 下层被遮挡，覆盖度门槛单独放宽
    far_min_support_px=100,
    far_slope_spread=0.02,  # 下层筋条斜率相对顶层斜率区间的允许外扩量
    min_bar_px=180,
    cross_margin_px=15,
    merge_px=35.0,            # 交叉点去重半径
    far_hough_min_len=60,     # 下层被上层遮挡、碎片化严重，段长阈值要放宽
    far_hough_thresh=40,
    support_reach_px=50,      # 交叉点的局部支撑检查范围
    support_min_ratio=0.55,   # 两个方向各需达到的支撑比例
    support_halfwidth=12,
    blob_inner_px=18,         # 实心块判据：四角取样框的内边界
    blob_outer_px=30,         #                   外边界
    blob_fill_limit=0.75,     #                   四角填充率上限
)


# --------------------------------------------------------------------------
# 骨架化（Zhang-Suen 细化）
# --------------------------------------------------------------------------
def zhang_suen(binary, max_iter=18):
    """Zhang-Suen 细化。输入二值掩膜，输出单像素宽骨架。"""
    img = (np.asarray(binary) > 0).astype(np.uint8)
    for _ in range(max_iter):
        removed_any = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            ring = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(ring[:-1])
            A = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
                    for i in range(8))
            if step == 0:
                cond = ((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0)
            else:
                cond = ((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0)
            kill = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if kill.any():
                img[kill] = 0
                removed_any = True
        if not removed_any:
            break
    return img


# --------------------------------------------------------------------------
# 按族参数化：横筋 y = a*x + b，竖筋 x = a*y + b
# --------------------------------------------------------------------------
def remove_junctions(skel):
    """去掉骨架的分叉点，把交叉成网格的骨架拆成一段段单筋曲线。

    网格骨架在每个交叉处会有 T 形/十字形分叉，HoughLinesP 会把分叉处的对角
    短线当成斜线，污染筋条聚类（表现为同一条竖筋被拆成多簇、或冒出斜率反号的
    伪筋条）。先去掉分叉点即可消除。
    """
    img = (np.asarray(skel) > 0).astype(np.uint8)
    p = np.pad(img, 1)
    nbr = sum(p[1 + dy:1 + dy + img.shape[0], 1 + dx:1 + dx + img.shape[1]]
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return (img > 0) & (nbr < 3)


def _segment_ab(seg):
    """把线段转成 (族, 斜率, 截距)。族为 'H' 或 'V'。"""
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) >= abs(dy):
        if abs(dx) < 1e-9:
            return None
        a = dy / dx
        return "H", a, y1 - a * x1
    if abs(dy) < 1e-9:
        return None
    a = dx / dy
    return "V", a, x1 - a * y1


def _point_to_line_dist(px, py, family, a, b):
    """点到族参数直线的垂直距离（像素）。"""
    if family == "H":
        return np.abs(py - a * px - b) / np.sqrt(1.0 + a * a)
    return np.abs(px - a * py - b) / np.sqrt(1.0 + a * a)


def _hough_segments(binary, params):
    lines = cv2.HoughLinesP(binary, 1, np.pi / 360,
                            threshold=params["hough_thresh"],
                            minLineLength=params["hough_min_len"],
                            maxLineGap=params["hough_max_gap"])
    if lines is None:
        return []
    # OpenCV 5 返回 (N, 4)，旧版返回 (N, 1, 4)，统一成 (N, 4)
    return [tuple(int(v) for v in row)
            for row in np.asarray(lines).reshape(-1, 4)]


def _cluster_ab(reps, params):
    """在单个族内按 (斜率, 截距) 聚类。reps 为 [(a, b)]。"""
    a_tol = params["cluster_slope_tol"]
    b_tol = params["cluster_intercept_px"]
    clusters = []
    for a, b in sorted(reps, key=lambda z: z[1]):
        for c in clusters:
            if abs(a - c[0]) < a_tol and abs(b - c[1]) < b_tol:
                k = c[2]
                c[0] = (c[0] * k + a) / (k + 1)
                c[1] = (c[1] * k + b) / (k + 1)
                c[2] = k + 1
                break
        else:
            clusters.append([a, b, 1])
    return clusters


def _refit(px, family, a, b, params):
    """对候选筋条附近的骨架像素做 PCA 精修。"""
    if px.shape[0] < 40:
        return None
    c = px.mean(0)
    _, sv, vt = np.linalg.svd(px - c, full_matrices=False)
    if sv[0] < 1e-9:
        return None
    direction = vt[0]
    proj = (px - c) @ direction
    perp = (px - c) @ vt[1]
    span = float(proj.max() - proj.min())
    # 筋条方向要与所属族一致
    ang = abs(np.degrees(np.arctan2(direction[1], direction[0])))
    if family == "H" and not (ang < 45.0 or ang > 135.0):
        return None
    if family == "V" and not (45.0 <= ang <= 135.0):
        return None
    return {"family": family, "point": c, "direction": direction,
            "span": span, "n_support": int(px.shape[0]),
            "straightness": float(np.abs(perp).max()),
            # 自检：单条筋的骨架支撑像素数应与跨度同量级；比值太小说明这条线是
            # 勉强连起来的弱线，比值明显大于 1 说明多条筋被并成了一簇
            "coverage": float(px.shape[0] / span) if span > 0 else float("inf"),
            "extent": (c + direction * proj.min(), c + direction * proj.max())}


def _ref_pos(line, shape):
    """筋条在图像中心处的横向（竖筋）/ 纵向（横筋）位置。

    判断两条候选线是不是同一根筋，不能直接比截距 b：b 是 y=0（或 x=0）处的
    截距，两条斜率略有差异的候选线在画面中部几乎重合，在原点处却可能差几十
    像素，于是同一条筋因为去重失败而变成两条。实测 station_8 左边缘那根竖筋
    就被拆成 b=39.6 与 b=68.6 两条：在 y=720 处只差 10px，按原点截距比较却差
    29px，超过去重阈值，于是多出一根幽灵筋、并多出一批幽灵交叉点。
    """
    a, b = line["ab"]
    if line["family"] == "H":
        return a * (shape[1] / 2.0) + b
    return a * (shape[0] / 2.0) + b


def _dedup(lines, params, shape):
    """合并同族内近乎重合的筋条（按图像中心处的位置比较，见 _ref_pos）。"""
    a_tol = params["dedup_slope_tol"]
    b_tol = params["dedup_intercept_px"]
    kept = []                       # [(筋条, 中心处位置)]
    for l in sorted(lines, key=lambda z: (z["family"], z["ab"][1])):
        a = l["ab"][0]
        r = _ref_pos(l, shape)
        hit = None
        for i, (k, kr) in enumerate(kept):
            if k["family"] != l["family"]:
                continue
            if abs(a - k["ab"][0]) < a_tol and abs(r - kr) < b_tol:
                hit = i
                break
        if hit is None:
            kept.append((l, r))
        elif l["n_support"] > kept[hit][0]["n_support"]:
            kept[hit] = (l, r)
    return [l for l, _ in kept]


def extract_bar_lines(binary, params=None):
    """从二值掩膜提取筋条轴线。返回 (lines, skeleton)。"""
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    k = p["close_ksize"]
    closed = cv2.morphologyEx((binary > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((k, k), np.uint8))
    skel = zhang_suen(closed, p["skel_max_iter"])
    branches = remove_junctions(skel)

    segments = _hough_segments(branches.astype(np.uint8) * 255, p)
    reps = {"H": [], "V": []}
    for s in segments:
        r = _segment_ab(s)
        if r is not None:
            reps[r[0]].append((r[1], r[2]))

    ys, xs = np.nonzero(branches)
    lines = []
    if ys.size:
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64)
        for family in ("H", "V"):
            for a, b, _ in _cluster_ab(reps[family], p):
                dist = _point_to_line_dist(xs, ys, family, a, b)
                sel = dist < p["refit_halfwidth"]
                if sel.sum() < p["min_support_px"]:
                    continue
                line = _refit(np.column_stack([xs[sel], ys[sel]]), family, a, b, p)
                if line is not None and line["coverage"] < p["min_coverage"]:
                    line = None          # 支撑覆盖太差，不是一根完整的筋
                if line is not None:
                    cx, cy = line["point"]
                    dx, dy = line["direction"]
                    if family == "H":
                        aa = float(dy / dx)
                        bb = float(cy - aa * cx)
                    else:
                        aa = float(dx / dy)
                        bb = float(cx - aa * cy)
                    line["ab"] = (aa, bb)
                    lines.append(line)
    return _dedup(lines, p, binary.shape), skel


def split_directions(lines):
    return ([l for l in lines if l["family"] == "H"],
            [l for l in lines if l["family"] == "V"])


# --------------------------------------------------------------------------
# 交叉点
# --------------------------------------------------------------------------
def intersect(l1, l2):
    d1, d2 = l1["direction"], l2["direction"]
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    dp = l2["point"] - l1["point"]
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    return l1["point"] + d1 * t


def _within(line, pt, margin):
    return abs(float((pt - line["point"]) @ line["direction"])) <= line["span"] / 2 + margin


def _slope_reference(lines, spread, min_coverage=0.9):
    """由一组筋条给出可接受的斜率区间。

    取"高质量筋条"（覆盖度 >= min_coverage）斜率的**中位数**再向两侧放开 spread。
    用中位数而不是 min/max：实测顶层里也偶尔混进一条 a≈0 的坏线，
    用极值会把允许区间拉得太宽，约束就失效了。
    """
    good = [l["ab"][0] for l in lines if l["coverage"] >= min_coverage]
    vals = good if good else [l["ab"][0] for l in lines]
    if not vals:
        return None
    c = float(np.median(vals))
    return c - spread, c + spread


def _filter_by_slope(lines, ref):
    if ref is None:
        return lines
    lo, hi = ref
    return [l for l in lines if lo <= l["ab"][0] <= hi]


def _direction_support(mask, pt, line, params):
    """沿筋条轴线正负两个方向检查掩膜是否连续跟随。
    返回 (正向比, 负向比)，某一侧“无法评估”时该侧为 None。
    两种情况算“无法评估”，都不能当成“没有支撑”：
      * 全部采样点落在图像外——实测 station_1 顶部的交叉点
        (131.5, 7.5) 就在第一行附近，向上采样必然全部越界；
      * 全部采样点超出该筋条实测跨度——交厹落在筋条端头时下一侧
        本来就没有钢筋，实测 station_9 右下角交叉点 (1048.6, 1397.3)
        就是这种情况（筋在交点下方 10px 处结束）。
    """
    direction = line["direction"]
    proj = [(e - line["point"]) @ direction for e in line["extent"]]
    lo, hi = min(proj), max(proj)
    # 交厹在轴线方向上的位置（相对筋条中心），采样点的绝对位置是 base + sign*t
    base = float((pt - line["point"]) @ direction)
    reach = params["support_reach_px"]
    half = params["support_halfwidth"]
    ratios = []
    for sign in (1.0, -1.0):
        hit = 0
        total = 0
        for t in np.linspace(10.0, reach, 9):
            s = sign * t
            if not lo <= base + s <= hi:     # 该处本来就没有这根筋，无从判断
                continue
            q = pt + direction * s
            x, y = int(round(q[0])), int(round(q[1]))
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                continue
            total += 1
            x0, x1 = max(0, x - half), min(mask.shape[1], x + half + 1)
            y0, y1 = max(0, y - half), min(mask.shape[0], y + half + 1)
            if mask[y0:y1, x0:x1].any():
                hit += 1
        ratios.append(hit / total if total else None)
    return ratios[0], ratios[1]


def crossing_supported(mask, pt, line_a, line_b, params):
    """交叉点是否真的落在两条连续的筋上（而不是外推出的假交点）。

    非钢筋杂物（圆形垫块、浅色长条）即使让某条直线勉强穿过，局部也不会有
    沿筋方向的连续支撑，据此判为"虚假干扰点"。

    只对“可评估”的方向（有采样点落在图像内）作要求，越界方向跳过。
    """
    need = params["support_min_ratio"]
    for line in (line_a, line_b):
        vals = [r for r in _direction_support(mask, pt, line, params)
                if r is not None]
        if vals and min(vals) < need:
            return False
    return True


def solid_around(mask, pt, params):
    """交叉点四周是否被掩膜整片填满（说明这里是实心块，不是两条细筋交叉）。

    真钢筋交叉：两条约一个钢筋直径宽的条带正交，四个斜角方向是空的。实测 294
    个保留点的四角填充率中位数为 0.00（九成以上恰好是 0）；而非钢筋的实心物体
    会把四角填满——实测 station_7 (86,545) 处为 1.00，且该处深度比它所依托的
    两条筋各近约 52mm，是一块挡在钢筋笼前面的实心件。
    """
    x, y = int(round(pt[0])), int(round(pt[1]))
    inner, outer = params["blob_inner_px"], params["blob_outer_px"]
    h, w = mask.shape
    fills = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = sorted((x + sx * inner, x + sx * outer))
            y0, y1 = sorted((y + sy * inner, y + sy * outer))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w), min(y1, h)
            if x1 > x0 and y1 > y0:
                fills.append(float(mask[y0:y1, x0:x1].mean()))
    if not fills:
        return False
    return float(np.mean(fills)) >= params["blob_fill_limit"]


def merge_points(points, radius):
    """合并距离小于 radius 的重复交叉点（重复筋条会给出成对的同一交点）。"""
    kept = []
    for item in sorted(points, key=lambda t: (-t[2] if np.isfinite(t[2]) else 0.0, t[0], t[1])):
        for k in kept:
            if (item[0] - k[0]) ** 2 + (item[1] - k[1]) ** 2 <= radius * radius:
                break
        else:
            kept.append(item)
    return sorted(kept, key=lambda t: (t[1], t[0]))


def local_depth(depth_mm, x, y, r=15):
    h, w = depth_mm.shape
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = depth_mm[y0:y1, x0:x1]
    v = patch[np.isfinite(patch) & (patch > 0)]
    if v.size < 10:
        return None
    return {"median": float(np.median(v)), "min": float(v.min()), "n": int(v.size)}


def detect_rebar_intersections(depth_mm, gray=None, params=None):
    """检测顶层交叉点，并给出被剔除的下层点与杂物点（返回 dict）。"""
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    split = auto_split_mm(depth_mm)
    if split is None:
        raise ValueError("无法自动分层，需要先检查深度直方图")
    top_mask, far_mask, _ = split_layers(depth_mm, split)

    # 下层被上层遮挡，筋条必然断续，覆盖度天然偏低（实测被挡住的竖筋只有 0.37），
    # 所以下层的覆盖度门槛和段长门槛都要单独放宽，否则会漏掉整条下层筋。
    far_params = dict(p)
    far_params["hough_min_len"] = p["far_hough_min_len"]
    far_params["hough_thresh"] = p["far_hough_thresh"]
    far_params["min_coverage"] = p["far_min_coverage"]
    far_params["min_support_px"] = p["far_min_support_px"]

    result = {"split_mm": float(split), "accepted": [], "rejected_lower": [],
              "rejected_clutter": [], "h_lines": [], "v_lines": [],
              "far_h_lines": [], "far_v_lines": [],
              "skeleton": None, "top_mask": top_mask, "far_mask": far_mask,
              "stats": {}}

    accepted, lower, clutter = [], [], []
    h_ref = v_ref = None
    for name, mask, layer_params in (("top", top_mask, p),
                                     ("far", far_mask, far_params)):
        lines, skel = extract_bar_lines(mask, layer_params)
        if name == "far":
            # 下层被上层遮挡后碎片化，Hough 会"架"出一些斜率明显不对的桥接线
            # （实测伪线斜率约 0，即几乎完美竖直，而真筋约 -0.057）。
            # 同一笼子的筋方向一致，所以用顶层测出的斜率区间来约束下层。
            spread = p["far_slope_spread"]
            lines = (_filter_by_slope([l for l in lines if l["family"] == "H"], h_ref)
                     + _filter_by_slope([l for l in lines if l["family"] == "V"], v_ref))
        horiz, vert = split_directions(lines)
        if name == "top":
            result["h_lines"], result["v_lines"] = horiz, vert
            result["skeleton"] = skel
            h_ref = _slope_reference(horiz, p["far_slope_spread"])
            v_ref = _slope_reference(vert, p["far_slope_spread"])
        else:
            result["far_h_lines"], result["far_v_lines"] = horiz, vert
        for a in horiz:
            for b in vert:
                pt = intersect(a, b)
                if pt is None:
                    continue
                x, y = int(round(pt[0])), int(round(pt[1]))
                if not (0 <= x < depth_mm.shape[1] and 0 <= y < depth_mm.shape[0]):
                    continue
                if not (_within(a, pt, p["cross_margin_px"])
                        and _within(b, pt, p["cross_margin_px"])):
                    continue
                d = local_depth(depth_mm, x, y)
                if d is None:
                    continue
                item = (x, y, d["median"])
                is_top = d["median"] < split
                short = min(a["span"], b["span"]) < p["min_bar_px"]
                if name == "far" or not is_top:
                    lower.append(item)          # 深度阈值筛掉的下层交叉点
                elif (short or not crossing_supported(mask, pt, a, b, p)
                      or solid_around(mask, pt, p)):
                    clutter.append(item)        # 形状/连续性不像钢筋
                else:
                    accepted.append(item)

    result["accepted"] = merge_points(accepted, p["merge_px"])
    result["rejected_lower"] = merge_points(lower, p["merge_px"])
    result["rejected_clutter"] = merge_points(clutter, p["merge_px"])
    merged = [(l["family"], round(l["coverage"], 2), int(round(l["span"])))
              for l in result["h_lines"] + result["v_lines"]
              if l["coverage"] > 1.8]
    result["stats"] = {"h_bars": len(result["h_lines"]),
                       "v_bars": len(result["v_lines"]),
                       "far_h_bars": len(result["far_h_lines"]),
                       "far_v_bars": len(result["far_v_lines"]),
                       "accepted": len(result["accepted"]),
                       "lower": len(result["rejected_lower"]),
                       "clutter": len(result["rejected_clutter"]),
                       "suspicious_merged": merged}
    return result


detect_intersections = detect_rebar_intersections
