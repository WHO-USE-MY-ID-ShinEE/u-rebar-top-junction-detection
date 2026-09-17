# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()

# 新增斜率参考的实现
helper = '''
def _slope_reference(lines, spread):
    """由一组筋条给出可接受的斜率区间。"""
    if not lines:
        return None
    vals = [l["ab"][0] for l in lines]
    return min(vals) - spread, max(vals) + spread


def _filter_by_slope(lines, ref):
    if ref is None:
        return lines
    lo, hi = ref
    return [l for l in lines if lo <= l["ab"][0] <= hi]


def _direction_support(mask, pt, direction, params):'''
assert "\ndef _direction_support(mask, pt, direction, params):" in s
s = s.replace("\ndef _direction_support(mask, pt, direction, params):", helper, 1)

# 参数：下层斜率允许在顶层斜率区间外再放宽多少
s = s.replace("    far_min_support_px=100,",
              "    far_min_support_px=100,\n"
              "    far_slope_spread=0.02,  # 下层筋条斜率相对顶层斜率区间的允许外扩量", 1)

# 主循环：先用顶层建立斜率参考，再用它约束下层
old = """    accepted, lower, clutter = [], [], []
    for name, mask, layer_params in (("top", top_mask, p),
                                     ("far", far_mask, far_params)):
        lines, skel = extract_bar_lines(mask, layer_params)
        horiz, vert = split_directions(lines)
        if name == "top":
            result["h_lines"], result["v_lines"] = horiz, vert
            result["skeleton"] = skel
        else:
            result["far_h_lines"], result["far_v_lines"] = horiz, vert"""
new = """    accepted, lower, clutter = [], [], []
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
            result["far_h_lines"], result["far_v_lines"] = horiz, vert"""
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
