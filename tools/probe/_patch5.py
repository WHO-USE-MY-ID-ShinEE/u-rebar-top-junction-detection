# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()

# 单条筋自检指标：支撑像素数 / 跨度，正常约等于 1
old = """                    line["ab"] = (aa, bb)
                    lines.append(line)"""
new = """                    line["ab"] = (aa, bb)
                    # 自检：单条筋的骨架支撑像素数应与跨度同量级，
                    # 比值明显大于 1 说明多条筋被并成了一簇
                    line["coverage"] = (line["n_support"] / line["span"]
                                        if line["span"] > 0 else float("inf"))
                    lines.append(line)"""
assert old in s
s = s.replace(old, new, 1)

# 在结果里汇总可疑筋条
old2 = """    result["stats"] = {"h_bars": len(result["h_lines"]),
                       "v_bars": len(result["v_lines"]),
                       "accepted": len(accepted), "lower": len(lower),
                       "clutter": len(clutter)}"""
new2 = """    merged = [(l["family"], round(l["coverage"], 2), int(round(l["span"])))
              for l in result["h_lines"] + result["v_lines"]
              if l["coverage"] > 1.8]
    result["stats"] = {"h_bars": len(result["h_lines"]),
                       "v_bars": len(result["v_lines"]),
                       "accepted": len(result["accepted"]),
                       "lower": len(result["rejected_lower"]),
                       "clutter": len(result["rejected_clutter"]),
                       "suspicious_merged": merged}"""
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
