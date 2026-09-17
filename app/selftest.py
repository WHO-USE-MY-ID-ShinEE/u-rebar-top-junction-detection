"""无界面自检：不依赖图形环境，验证"加载 -> 检测 -> 审计 -> 导出"整条链路。

自检内容：
  1. 合成图像对：4 条顶层横筋 x 3 条顶层竖筋 = 12 个交叉点；另有 1 条带断口的
     顶层竖筋（断口处的交点局部支撑不足，应判为虚假干扰点，另一处交点为正常点）、
     1 条下层横筋 + 1 条下层竖筋（交点应判为下层点）、1 块实心杂物（不应产生交叉点）。
  2. 同一对图像的深度图换一种存放方式（与灰度图互为转置，需自动旋转对齐），
     结果应与第 1 步逐点相同。
  3. 数据集工位：站号与保留点数应与历史一致；若 outputs/ 下有历史产物，
     逐字节比对导出的坐标文件，确保界面/自测路径与命令行流水线完全一致。

用法：python app.py --selftest   （等价于 python -m app.selftest）
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src import imgio, pipeline

from . import runner

W, H = 1000, 800
TOP_M, FAR_M = 0.750, 0.900          # 顶层 / 下层深度（米）
H_BARS = (150, 350, 550, 750)        # 顶层横筋的 y
V_BARS = (200, 450, 700)             # 顶层竖筋的 x
BROKEN_V = 580                       # 带断口的竖筋：断口处的交点应判为虚假干扰点
BROKEN_SPAN = (250, 650)             #   该竖筋的 y 范围
BROKEN_GAP = (495, 605)              #   断开的一段（模拟被杂物压住 / 折断）
FAR_H = (250, 100, 900)              # 下层横筋：y, x0, x1
FAR_V = (850, 80, 780)               # 下层竖筋：x, y0, y1
BLOCK = (120, 620, 60)               # 实心杂物：圆心 x, y, 半径
BAR_W = 15                           # 取奇数宽度，让筋条中心正好落在整数像素上

EXPECTED_TOP = sorted([(x, y) for x in V_BARS for y in H_BARS] + [(BROKEN_V, 350)])
EXPECTED_LOWER = (FAR_V[0], FAR_H[0])
EXPECTED_CLUTTER = (BROKEN_V, 550)


def synth_pair():
    """造一对合成图像，返回 (gray, depth_m)。"""
    rng = np.random.default_rng(20260918)
    gray = rng.integers(4, 14, size=(H, W), dtype=np.uint8)
    depth = np.full((H, W), np.nan, np.float32)

    def bar_h(y, x0, x1, value):
        gray[y - BAR_W // 2:y + BAR_W // 2 + 1, x0:x1] = 200
        depth[y - BAR_W // 2:y + BAR_W // 2 + 1, x0:x1] = value

    def bar_v(x, y0, y1, value):
        gray[y0:y1, x - BAR_W // 2:x + BAR_W // 2 + 1] = 200
        depth[y0:y1, x - BAR_W // 2:x + BAR_W // 2 + 1] = value

    # 先画杂物与下层，再画顶层：交叉处顶层把下层覆盖掉（等于相机的遮挡关系）
    cx, cy, r = BLOCK
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    gray[mask > 0] = 180
    depth[mask > 0] = TOP_M + 0.01

    bar_h(FAR_H[0], FAR_H[1], FAR_H[2], FAR_M)
    bar_v(FAR_V[0], FAR_V[1], FAR_V[2], FAR_M)

    for y in H_BARS:
        bar_h(y, 100, 900, TOP_M)
    for x in V_BARS:
        bar_v(x, 80, 780, TOP_M)
    # 带断口的竖筋：断口正好压在 y=550 那条横筋上，此处局部支撑不足
    bar_v(BROKEN_V, BROKEN_SPAN[0], BROKEN_GAP[0], TOP_M)
    bar_v(BROKEN_V, BROKEN_GAP[1], BROKEN_SPAN[1], TOP_M)
    return gray, depth


def _write_pair(dir_path, gray, depth, transposed=False):
    """写出一个图像对，返回 (深度图路径, 灰度图路径)。

    transposed=True 时按数据集的真实存放方式写：深度图与灰度图互为转置，
    加载时必须自动逆时针旋转 90 度才能对齐。
    """
    dir_path = Path(dir_path)
    gray_p = dir_path / "synth_output_image_left.png"
    depth_p = dir_path / "synth_output_depth_raw.tif"
    imgio.imwrite_any(gray_p, gray)
    raw = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE) if transposed else depth
    imgio.imwrite_any(depth_p, raw)
    return depth_p, gray_p


def check_synth(tmp, results):
    """合成图像对：点数、三类点的判归、导出文件。"""
    gray, depth = synth_pair()
    for transposed in (False, True):
        tag = "转置存放" if transposed else "同向存放"
        depth_p, gray_p = _write_pair(Path(tmp) / ("t" if transposed else "n"),
                                      gray, depth, transposed)
        state = runner.analyze(depth_path=str(depth_p), gray_path=str(gray_p),
                               params={"split_mm": 830.0})
        got = sorted((int(x), int(y)) for x, y, _ in state["res"]["accepted"])
        results.append((f"合成图像对({tag})：顶层交叉点 {len(EXPECTED_TOP)} 个",
                        _match(got, EXPECTED_TOP),
                        f"实得 {len(got)} 个: {got}"))
        lower = [(int(x), int(y)) for x, y, _ in state["res"]["rejected_lower"]]
        results.append((f"合成图像对({tag})：下层交叉点被判为下层",
                        len(lower) == 1 and _near(lower[0], EXPECTED_LOWER),
                        f"实得 {lower}"))
        clutter = [(int(x), int(y)) for x, y, _ in state["res"]["rejected_clutter"]]
        results.append((f"合成图像对({tag})：断口处的交点被判为干扰",
                        any(_near(c, EXPECTED_CLUTTER) for c in clutter),
                        f"实得 {clutter}"))
        results.append((f"合成图像对({tag})：无误检、无真漏检",
                        len(state["audit"]["extra"]) == 0
                        and state["audit"]["lost"] == 0,
                        f"多余 {len(state['audit']['extra'])} 个，真漏检 "
                        f"{state['audit']['lost']} 个"))
        if not transposed:
            out = Path(tmp) / "out"
            files = runner.export(state, out)
            csv_lines = Path(files["csv"]).read_text(encoding="utf-8").strip().split("\n")
            pts = json.loads(Path(files["json"]).read_text(encoding="utf-8"))["points"]
            results.append(("合成图像对：导出 csv/json/结果图齐全",
                            all(Path(v).exists() for v in files.values()),
                            str(sorted(files))))
            results.append(("合成图像对：导出内容与检测点数一致",
                            len(csv_lines) == len(EXPECTED_TOP) + 1
                            and len(pts) == len(EXPECTED_TOP),
                            f"csv {len(csv_lines)} 行，json {len(pts)} 点"))


def check_dataset(tmp, results):
    """数据集工位：与历史产物逐字节比对（有数据才跑）。"""
    data_dir = Path(pipeline.DEFAULT_DATA_DIR)
    stations = imgio.list_stations(data_dir) if data_dir.exists() else []
    if not stations:
        results.append(("数据集工位：跳过", True, f"未找到数据目录 {data_dir}"))
        return
    out_dir = Path(pipeline.DEFAULT_OUT_DIR)
    for n in stations:
        state = runner.analyze(data_dir=str(data_dir), station=n)
        ref = None
        if (out_dir / f"station_{n}_points.json").exists():
            ref = out_dir / f"station_{n}_points.json"
        count = state["stats"]["accepted"]
        results.append((f"station_{n}：保留点数 {count}",
                        ref is None or count == _ref_count(ref),
                        f"期望 {_ref_count(ref) if ref else '无历史产物'}"))

    n = stations[0]
    if (out_dir / f"station_{n}_points.json").exists():
        state = runner.analyze(data_dir=str(data_dir), station=n)
        files = runner.export(state, Path(tmp) / "ds")
        same = []
        for key, name in (("csv", f"station_{n}_points.csv"),
                          ("json", f"station_{n}_points.json"),
                          ("rejected", f"station_{n}_rejected.json"),
                          ("result_png", f"station_{n}_result.png")):
            a = Path(files[key]).read_bytes()
            b = (out_dir / name).read_bytes()
            same.append((key, a == b))
        ok = all(v for _, v in same)
        results.append((f"station_{n}：导出与命令行流水线逐字节一致", ok, str(same)))


def _ref_count(path):
    return len(json.loads(Path(path).read_text(encoding="utf-8"))["points"])


def _near(pt, target, tol=20):
    return abs(pt[0] - target[0]) <= tol and abs(pt[1] - target[1]) <= tol


def _match(got, expected, tol=2):
    """两组点几何比对，允许 ±tol 像素的取整误差。"""
    if len(got) != len(expected):
        return False
    rest = list(expected)
    for pt in got:
        for i, item in enumerate(rest):
            if _near(pt, item, tol):
                rest.pop(i)
                break
        else:
            return False
    return not rest


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="gygj_selftest_") as tmp:
        check_synth(tmp, results)
        check_dataset(tmp, results)

    print("=" * 72)
    print("课题四 检测应用 · 无界面自检")
    print("=" * 72)
    failed = 0
    for label, ok, detail in results:
        print(f"[{'通过' if ok else '失败'}] {label}")
        if not ok:
            failed += 1
            print(f"        {detail}")
    print("-" * 72)
    print(f"共 {len(results)} 项，失败 {failed} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
