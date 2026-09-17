"""交叉点完备性审计（开发用探针）。

检测算法本身是"顶层横筋 x 顶层竖筋两两求交 -> 逐个判定是否保留"。本脚本反向检查：
把几何上真正落在两条筋条跨度之内的交点全部列成"期望交叉点"，与最终保留点比对，
回答三个问题：

  1. 期望交叉点有多少、保留了多少（完备性）；
  2. 被丢掉的那些分别是什么原因（深度判为下层 / 筋条过短 / 两向支撑不足 / 无回波）；
  3. 有没有"判为保留却没出现在结果里"的点（说明被去重合并吃掉了，属于可疑）。

用法：
    python tools/probe/audit_junctions.py             # 全站审计表
    python tools/probe/audit_junctions.py --sheet     # 另存待复查点的放大图
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, enhance, imgio, pipeline, viz  # noqa: E402

MATCH_PX = 20.0          # 期望点与保留点算作同一个点的距离
HALF = 130               # 复查图裁剪半径
TILE = 260               # 复查图每块的正方形边长


def _nearest(pt, pts):
    if not pts:
        return None
    arr = np.asarray(pts, float)
    d = np.hypot(arr[:, 0] - pt[0], arr[:, 1] - pt[1])
    return float(d.min())


def audit_station(station, data_dir):
    """返回该站的期望交叉点、吻合数、缺失点、多余点与缺失原因统计。"""
    gray, depth_mm, _ = imgio.load_station(data_dir, station)
    res = detect.detect_rebar_intersections(depth_mm, gray)
    p = detect.DEFAULTS
    split = res["split_mm"]
    accepted = [(x, y) for x, y, _ in res["accepted"]]

    rows = []
    for a in res["h_lines"]:
        for b in res["v_lines"]:
            pt = detect.intersect(a, b)
            if pt is None:
                continue
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            if not (0 <= x < depth_mm.shape[1] and 0 <= y < depth_mm.shape[0]):
                continue
            if not (detect._within(a, pt, p["cross_margin_px"])
                    and detect._within(b, pt, p["cross_margin_px"])):
                continue
            d = detect.local_depth(depth_mm, x, y)
            if d is None:
                reason = "无深度回波"
            elif d["median"] >= split:
                reason = "深度判为下层"
            elif min(a["span"], b["span"]) < p["min_bar_px"]:
                reason = "筋条过短"
            elif not detect.crossing_supported(res["top_mask"], pt, a, b, p):
                reason = "两向支撑不足"
            elif detect.solid_around(res["top_mask"], pt, p):
                reason = "压在实心杂物上"
            else:
                reason = "保留"
            rows.append({"x": x, "y": y, "reason": reason,
                         "dist": _nearest((x, y), accepted)})

    missing = [r for r in rows if r["dist"] is None or r["dist"] > MATCH_PX]
    # 判为保留、却找不到对应保留点的，说明点在去重合并里被吃掉了，值得警惕
    lost = [r for r in missing if r["reason"] == "保留"]

    extra = []
    exp = np.asarray([(r["x"], r["y"]) for r in rows], float)
    for pt in accepted:
        if exp.size == 0 or np.hypot(exp[:, 0] - pt[0],
                                     exp[:, 1] - pt[1]).min() > MATCH_PX:
            extra.append(pt)

    return {
        "station": station,
        "h_bars": len(res["h_lines"]),
        "v_bars": len(res["v_lines"]),
        "expected": len(rows),
        "matched": len(rows) - len(missing),
        "missing": missing,
        "lost": lost,
        "extra": extra,
        "accepted": len(accepted),
        "accepted_pts": accepted,
        "reasons": Counter(r["reason"] for r in missing),
    }


def _tile(base, x, y, marked):
    """从 base 上以 (x,y) 为中心裁一块固定大小的方块，并画十字与标记点。"""
    base = viz.to_bgr(base)
    h, w = base.shape[:2]
    canvas = np.full((TILE, TILE, 3), 255, np.uint8)
    x0, x1 = max(0, x - HALF), min(w, x + HALF)
    y0, y1 = max(0, y - HALF), min(h, y + HALF)
    canvas[y0 - (y - HALF):y1 - (y - HALF), x0 - (x - HALF):x1 - (x - HALF)] = \
        base[y0:y1, x0:x1]
    cx = TILE // 2
    cv2.drawMarker(canvas, (cx, cx), (255, 255, 255), cv2.MARKER_CROSS, 34, 4)
    cv2.drawMarker(canvas, (cx, cx), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
    for px, py in marked:                      # 周边已保留的点
        if abs(px - x) <= HALF and abs(py - y) <= HALF:
            cv2.circle(canvas, (px - (x - HALF), py - (y - HALF)), 8,
                       (0, 200, 0), 2)
    return canvas


def render_sheet(flagged, data_dir, out_path):
    """把待复查的点裁成一页图：左=增强灰度，右=深度伪彩。"""
    rows = []
    cache = {}
    for st, x, y, reason in flagged:
        if st not in cache:
            gray, depth_mm, _ = imgio.load_station(data_dir, st)
            gray_enh, _ = enhance.enhance(gray)
            res = detect.detect_rebar_intersections(depth_mm, gray)
            cache[st] = (gray_enh, enhance.depth_visual(depth_mm),
                         [(px, py) for px, py, _ in res["accepted"]])
        gray_enh, depth_vis, accepted = cache[st]
        g = _tile(gray_enh, x, y, [(px, py) for px, py in accepted
                                   if abs(px - x) <= HALF and abs(py - y) <= HALF])
        d = _tile(depth_vis, x, y, [])
        row = np.full((TILE + 46, TILE * 2 + 18, 3), 255, np.uint8)
        row[:TILE, :TILE] = g
        row[:TILE, TILE + 18:TILE * 2 + 18] = d
        viz.draw_text(row, f"station_{st}  ({x},{y})  {reason}",
                      (6, TILE + 4), 26, (0, 0, 0))
        rows.append(row)
    if not rows:
        print("没有需要复查的点")
        return
    sheet = np.vstack(rows)
    cv2.imencode(".png", sheet)[1].tofile(str(out_path))
    print(f"复查图已保存：{out_path}")


def main(argv):
    data_dir = pipeline.DEFAULT_DATA_DIR
    want_sheet = "--sheet" in argv
    argv = [a for a in argv if a != "--sheet"]
    stations = [int(s) for s in argv] or imgio.list_stations(data_dir)
    hdr = (f"{'站':>3} {'横筋':>4} {'竖筋':>4} {'期望':>5} {'吻合':>5} "
           f"{'漏检':>5} {'多余':>5}  漏检原因")
    print(hdr)
    print("-" * 64)
    tot = dict(expected=0, matched=0, missing=0, accepted=0, extra=0, lost=0)
    anomalies = []
    for st in stations:
        r = audit_station(st, data_dir)
        tot["expected"] += r["expected"]
        tot["matched"] += r["matched"]
        tot["missing"] += len(r["missing"])
        tot["accepted"] += r["accepted"]
        tot["extra"] += len(r["extra"])
        tot["lost"] += len(r["lost"])
        reason_txt = " ".join(f"{k}={v}" for k, v in r["reasons"].items() if v)
        print(f"{st:>3} {r['h_bars']:>4} {r['v_bars']:>4} {r['expected']:>5} "
              f"{r['matched']:>5} {len(r['missing']):>5} {len(r['extra']):>5}  "
              f"{reason_txt or '--'}")
        for q in r["missing"]:
            anomalies.append((st, q["x"], q["y"], q["reason"]))
    print("-" * 64)
    rate = tot["matched"] / tot["expected"] if tot["expected"] else 0.0
    print(f"合计：期望 {tot['expected']}，吻合 {tot['matched']}，"
          f"漏掉 {tot['missing']}（完备率 {rate:.1%}），"
          f"保留 {tot['accepted']}，其中多余 {tot['extra']}，可疑丢失 {tot['lost']}")
    if anomalies:
        print("\n需要人眼复查的点：")
        order = ["保留", "两向支撑不足", "筋条过短", "深度判为下层", "无深度回波"]
        for st, x, y, reason in sorted(
                anomalies, key=lambda z: (order.index(z[3])
                                          if z[3] in order else 99, z[0])):
            print(f"  station_{st}  ({x:>4}, {y:>4})  {reason}")
    if want_sheet:
        render_sheet(anomalies, data_dir,
                     ROOT / "tools" / "probe" / "preview" / "audit_review.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
