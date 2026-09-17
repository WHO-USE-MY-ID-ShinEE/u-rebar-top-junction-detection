"""交叉点完备性审计（命令行入口，逻辑在 src/audit.py）。

用法：
    python tools/probe/audit_junctions.py              # 全站审计表
    python tools/probe/audit_junctions.py --sheet      # 另存待复查点的放大图
    python tools/probe/audit_junctions.py 1 6          # 只跑指定工位
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import audit, detect, enhance, imgio, pipeline, viz  # noqa: E402

HALF = 130               # 复查图裁剪半径
TILE = 260               # 复查图每块的正方形边长


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
        cv2.circle(canvas, (px - (x - HALF), py - (y - HALF)), 8, (0, 200, 0), 2)
    return canvas


def render_sheet(flagged, data_dir, out_path):
    """把待复查的点裁成一页图：左=增强灰度，右=深度伪彩。"""
    rows, cache = [], {}
    for st, x, y, reason in flagged:
        if st not in cache:
            gray, depth_mm, _ = imgio.load_station(data_dir, st)
            gray_enh, _ = enhance.enhance(gray)
            res = detect.detect_rebar_intersections(depth_mm, gray_enh)
            cache[st] = (gray_enh, enhance.depth_visual(depth_mm),
                         [(px, py) for px, py, _ in res["accepted"]])
        gray_enh, depth_vis, accepted = cache[st]
        near = [(px, py) for px, py in accepted
                if abs(px - x) <= HALF and abs(py - y) <= HALF]
        row = np.full((TILE + 46, TILE * 2 + 18, 3), 255, np.uint8)
        row[:TILE, :TILE] = _tile(gray_enh, x, y, near)
        row[:TILE, TILE + 18:TILE * 2 + 18] = _tile(depth_vis, x, y, [])
        viz.draw_text(row, f"station_{st}  ({x},{y})  {reason}",
                      (6, TILE + 4), 26, (0, 0, 0))
        rows.append(row)
    if not rows:
        print("没有需要复查的点")
        return
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(out_path))
    print(f"复查图已保存：{out_path}")


def main(argv):
    data_dir = pipeline.DEFAULT_DATA_DIR
    want_sheet = "--sheet" in argv
    argv = [a for a in argv if a != "--sheet"]
    stations = [int(s) for s in argv] or imgio.list_stations(data_dir)

    hdr = (f"{'站':>3} {'横筋':>4} {'竖筋':>4} {'期望':>5} {'吻合':>5} "
           f"{'漏检':>5} {'多余':>5}  未保留原因")
    print(hdr)
    print("-" * 64)
    tot = dict(expected=0, matched=0, accepted=0, extra=0)
    anomalies = []
    for st in stations:
        res, rep = audit.audit_station(st, data_dir)
        tot["expected"] += rep["expected"]
        tot["matched"] += rep["matched"]
        tot["accepted"] += rep["accepted"]
        tot["extra"] += len(rep["extra"])
        reason_txt = " ".join(f"{k}={v}" for k, v in rep["reasons"].items() if v)
        print(f"{st:>3} {res['stats']['h_bars']:>4} {res['stats']['v_bars']:>4} "
              f"{rep['expected']:>5} {rep['matched']:>5} "
              f"{rep['expected'] - rep['matched']:>5} {len(rep['extra']):>5}  "
              f"{reason_txt or '--'}")
        for q in rep["missing"]:
            anomalies.append((st, q["x"], q["y"], q["reason"]))
    print("-" * 64)
    rate = tot["matched"] / tot["expected"] if tot["expected"] else 0.0
    print(f"合计：期望 {tot['expected']}，吻合 {tot['matched']}，"
          f"漏掉 {tot['expected'] - tot['matched']}（完备率 {rate:.1%}），"
          f"保留 {tot['accepted']}，其中多余 {tot['extra']}")
    if anomalies:
        print("\n未被保留的期望交叉点（逐个人工复核）：")
        order = list(audit.REASON_ORDER)
        for st, x, y, reason in sorted(
                anomalies, key=lambda z: (order.index(z[3]) if z[3] in order else 99,
                                          z[0])):
            print(f"  station_{st}  ({x:>4}, {y:>4})  {reason}")
    if want_sheet:
        render_sheet(anomalies, data_dir,
                     ROOT / "tools" / "probe" / "preview" / "audit_review.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
