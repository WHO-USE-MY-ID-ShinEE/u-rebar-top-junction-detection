# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()
old = '''def _slope_reference(lines, spread):
    """由一组筋条给出可接受的斜率区间。"""
    if not lines:
        return None
    vals = [l["ab"][0] for l in lines]
    return min(vals) - spread, max(vals) + spread'''
new = '''def _slope_reference(lines, spread, min_coverage=0.9):
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
    return c - spread, c + spread'''
assert old in s
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched")
