# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()
old = """                if line is not None:
                    line["ab"] = (float(line["direction"][1] / line["direction"][0])
                                  if family == "H"
                                  else float(line["direction"][0] / line["direction"][1]))
                    lines.append(line)"""
new = """                if line is not None:
                    cx, cy = line["point"]
                    dx, dy = line["direction"]
                    if family == "H":
                        aa = float(dy / dx)
                        bb = float(cy - aa * cx)
                    else:
                        aa = float(dx / dy)
                        bb = float(cx - aa * cy)
                    line["ab"] = (aa, bb)
                    lines.append(line)"""
assert old in s, "pattern not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
