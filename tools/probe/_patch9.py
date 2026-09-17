# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\pipeline.py"
s = io.open(p, encoding="utf-8").read()
old = """        json.dump({"station": station,
                   "size_wh": [int(result["h_lines"][0]["point"].shape[0]) if False else 0],
                   "points": [[int(x), int(y)] for x, y, _ in points]},
                  f, ensure_ascii=False, indent=1)"""
new = """        json.dump({"station": station,
                   "points": [[int(x), int(y)] for x, y, _ in points]},
                  f, ensure_ascii=False, indent=1)"""
assert old in s
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("cleaned")
