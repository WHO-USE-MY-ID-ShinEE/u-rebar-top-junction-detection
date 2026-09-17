# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace("    min_support_px=120,",
              "    min_support_px=120,\n    min_coverage=0.5,       # 支撑像素数/跨度 的下限，滤掉勉强连起来的弱线", 1)
old = """                line = _refit(np.column_stack([xs[sel], ys[sel]]), family, a, b, p)
                if line is not None:"""
new = """                line = _refit(np.column_stack([xs[sel], ys[sel]]), family, a, b, p)
                if line is not None and line["coverage"] < p["min_coverage"]:
                    line = None          # 支撑覆盖太差，不是一根完整的筋
                if line is not None:"""
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
