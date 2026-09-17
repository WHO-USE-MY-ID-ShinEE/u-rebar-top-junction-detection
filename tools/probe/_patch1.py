# -*- coding: utf-8 -*-
import re, io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()
old = """    if lines is None:
        return []
    return [tuple(int(v) for v in l[0]) for l in lines]"""
new = """    if lines is None:
        return []
    # OpenCV 5 返回 (N, 4)，旧版返回 (N, 1, 4)，统一成 (N, 4)
    return [tuple(int(v) for v in row)
            for row in np.asarray(lines).reshape(-1, 4)]"""
assert old in s, "pattern not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
