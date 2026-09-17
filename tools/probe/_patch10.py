# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()

# 下层被上层遮挡，覆盖度天然偏低，不能用顶层的标准
old = """    far_params = dict(p)
    far_params["hough_min_len"] = p["far_hough_min_len"]
    far_params["hough_thresh"] = p["far_hough_thresh"]"""
new = """    # 下层被上层遮挡，筋条必然断续，覆盖度天然偏低（实测被挡住的竖筋只有 0.37），
    # 所以下层的覆盖度门槛和段长门槛都要单独放宽，否则会漏掉整条下层筋。
    far_params = dict(p)
    far_params["hough_min_len"] = p["far_hough_min_len"]
    far_params["hough_thresh"] = p["far_hough_thresh"]
    far_params["min_coverage"] = p["far_min_coverage"]
    far_params["min_support_px"] = p["far_min_support_px"]"""
assert old in s
s = s.replace(old, new, 1)

s = s.replace("    min_coverage=0.5,       # 支撑像素数/跨度 的下限，滤掉勉强连起来的弱线",
              "    min_coverage=0.5,       # 支撑像素数/跨度 的下限，滤掉勉强连起来的弱线\n"
              "    far_min_coverage=0.2,   # 下层被遮挡，覆盖度门槛单独放宽\n"
              "    far_min_support_px=100,", 1)

# 统计里补上下层筋条数，便于核对
old2 = """        if name == "top":
            result["h_lines"], result["v_lines"] = horiz, vert
            result["skeleton"] = skel"""
new2 = """        if name == "top":
            result["h_lines"], result["v_lines"] = horiz, vert
            result["skeleton"] = skel
        else:
            result["far_h_lines"], result["far_v_lines"] = horiz, vert"""
assert old2 in s
s = s.replace(old2, new2, 1)

s = s.replace("""    result = {"split_mm": float(split), "accepted": [], "rejected_lower": [],
              "rejected_clutter": [], "h_lines": [], "v_lines": [],
              "skeleton": None, "top_mask": top_mask, "far_mask": far_mask,
              "stats": {}}""",
"""    result = {"split_mm": float(split), "accepted": [], "rejected_lower": [],
              "rejected_clutter": [], "h_lines": [], "v_lines": [],
              "far_h_lines": [], "far_v_lines": [],
              "skeleton": None, "top_mask": top_mask, "far_mask": far_mask,
              "stats": {}}""", 1)

s = s.replace("""    result["stats"] = {"h_bars": len(result["h_lines"]),
                       "v_bars": len(result["v_lines"]),""",
"""    result["stats"] = {"h_bars": len(result["h_lines"]),
                       "v_bars": len(result["v_lines"]),
                       "far_h_bars": len(result["far_h_lines"]),
                       "far_v_bars": len(result["far_v_lines"]),""", 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
