# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()

# 把 coverage 的计算挪进 _refit，保证在过滤之前就有值
old = """    return {"family": family, "point": c, "direction": direction,
            "span": span, "n_support": int(px.shape[0]),
            "straightness": float(np.abs(perp).max()),
            "extent": (c + direction * proj.min(), c + direction * proj.max())}"""
new = """    return {"family": family, "point": c, "direction": direction,
            "span": span, "n_support": int(px.shape[0]),
            "straightness": float(np.abs(perp).max()),
            # 自检：单条筋的骨架支撑像素数应与跨度同量级；比值太小说明这条线是
            # 勉强连起来的弱线，比值明显大于 1 说明多条筋被并成了一簇
            "coverage": float(px.shape[0] / span) if span > 0 else float("inf"),
            "extent": (c + direction * proj.min(), c + direction * proj.max())}"""
assert old in s
s = s.replace(old, new, 1)

old2 = """                    line["ab"] = (aa, bb)
                    # 自检：单条筋的骨架支撑像素数应与跨度同量级，
                    # 比值明显大于 1 说明多条筋被并成了一簇
                    line["coverage"] = (line["n_support"] / line["span"]
                                        if line["span"] > 0 else float("inf"))
                    lines.append(line)"""
new2 = """                    line["ab"] = (aa, bb)
                    lines.append(line)"""
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
