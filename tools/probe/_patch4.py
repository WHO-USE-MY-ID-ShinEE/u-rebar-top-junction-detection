# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()

# ---- 1) 新增参数 ----
s = s.replace(
"""    min_support_px=120,
    min_bar_px=180,
    cross_margin_px=15,
)""",
"""    min_support_px=120,
    min_bar_px=180,
    cross_margin_px=15,
    merge_px=35.0,            # 交叉点去重半径
    far_hough_min_len=60,     # 下层被上层遮挡、碎片化严重，段长阈值要放宽
    far_hough_thresh=40,
    support_reach_px=50,      # 交叉点的局部支撑检查范围
    support_min_ratio=0.55,   # 两个方向各需达到的支撑比例
    support_halfwidth=12,
)""", 1)
assert "merge_px" in s

# ---- 2) 新增：交叉点局部支撑检查 + 去重 ----
helper = '''
def _direction_support(mask, pt, direction, params):
    """沿 direction 正负两个方向检查掩膜是否连续跟随。返回 (正向比, 负向比)。"""
    reach = params["support_reach_px"]
    half = params["support_halfwidth"]
    normal = np.array([-direction[1], direction[0]])
    ratios = []
    for sign in (1.0, -1.0):
        hit = 0
        total = 0
        for t in np.linspace(10.0, reach, 9):
            q = pt + direction * (sign * t)
            x, y = int(round(q[0])), int(round(q[1]))
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                continue
            total += 1
            x0, x1 = max(0, x - half), min(mask.shape[1], x + half + 1)
            y0, y1 = max(0, y - half), min(mask.shape[0], y + half + 1)
            if mask[y0:y1, x0:x1].any():
                hit += 1
        ratios.append(hit / total if total else 0.0)
    return ratios[0], ratios[1]


def crossing_supported(mask, pt, line_a, line_b, params):
    """交叉点是否真的落在两条连续的筋上（而不是外推出的假交点）。

    非钢筋杂物（圆形垫块、浅色长条）即使让某条直线勉强穿过，局部也不会有
    沿筋方向的连续支撑，据此判为"虚假干扰点"。
    """
    need = params["support_min_ratio"]
    for line in (line_a, line_b):
        pos, neg = _direction_support(mask, pt, line["direction"], params)
        if pos < need or neg < need:
            return False
    return True


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


def local_depth(depth_mm, x, y, r=15):'''
s = s.replace("\ndef local_depth(depth_mm, x, y, r=15):", helper, 1)

# ---- 3) 重写分类逻辑 ----
old = """    accepted, lower, clutter = [], [], []
    for name, mask in (("top", top_mask), ("far", far_mask)):
        lines, skel = extract_bar_lines(mask, p)
        horiz, vert = split_directions(lines)
        if name == "top":
            result["h_lines"], result["v_lines"] = horiz, vert
            result["skeleton"] = skel
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
                short = min(a["span"], b["span"]) < p["min_bar_px"]
                if name == "far":
                    lower.append(item)
                elif short:
                    clutter.append(item)
                else:
                    accepted.append(item)

    result["accepted"] = accepted
    result["rejected_lower"] = lower
    result["rejected_clutter"] = clutter"""
new = """    accepted, lower, clutter = [], [], []
    for name, mask, layer_params in (("top", top_mask, p),
                                     ("far", far_mask, far_params)):
        lines, skel = extract_bar_lines(mask, layer_params)
        horiz, vert = split_directions(lines)
        if name == "top":
            result["h_lines"], result["v_lines"] = horiz, vert
            result["skeleton"] = skel
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
                elif short or not crossing_supported(mask, pt, a, b, p):
                    clutter.append(item)        # 形状/连续性不像钢筋
                else:
                    accepted.append(item)

    result["accepted"] = merge_points(accepted, p["merge_px"])
    result["rejected_lower"] = merge_points(lower, p["merge_px"])
    result["rejected_clutter"] = merge_points(clutter, p["merge_px"])"""
assert old in s
s = s.replace(old, new, 1)

# ---- 4) 构造 far_params ----
old2 = """    result = {"split_mm": float(split), "accepted": [], "rejected_lower": [],"""
new2 = """    far_params = dict(p)
    far_params["hough_min_len"] = p["far_hough_min_len"]
    far_params["hough_thresh"] = p["far_hough_thresh"]

    result = {"split_mm": float(split), "accepted": [], "rejected_lower": [],"""
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("detect.py patched")
