"""对比两次运行（例如 Windows 开发端与 Ubuntu 验证端）的交叉点输出是否一致。

用法：
    python tools/verify_outputs.py                     # ref_win/ 对比 outputs/
    python tools/verify_outputs.py 参考目录 待比较目录

比较内容：
  * 每个工位的 station_N_points.csv：点数、像素坐标是否逐个相同、深度值最大偏差；
  * summary.csv：深度分界（容差 0.1mm）与其余各项整数是否相同。
像素坐标是整数，只要算法确定性成立就应当完全一致；深度值是浮点中位数，
允许极小偏差，因此单独报告最大偏差。
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Z_TOL = 0.05          # 深度值允许的最大偏差（mm）
SPLIT_TOL = 0.1       # 深度分界允许的最大偏差（mm）


def read_points(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [(int(r["x_px"]), int(r["y_px"]), float(r["z_mm"])) for r in rows]


def read_summary(path):
    # summary.csv 带 BOM（utf-8-sig，便于 Excel 直接打开），这里要按带 BOM 读
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {int(r["station"]): r for r in csv.DictReader(fh)}


def main(argv):
    ref_dir = Path(argv[0]) if argv else ROOT / "ref_win"
    test_dir = Path(argv[1]) if len(argv) > 1 else ROOT / "outputs"
    if not ref_dir.is_dir() or not test_dir.is_dir():
        sys.exit("\u76ee\u5f55\u4e0d\u5b58\u5728\uff1a%s / %s" % (ref_dir, test_dir))

    stations = sorted(int(p.name.split("_")[1])
                      for p in ref_dir.glob("station_*_points.csv"))
    bad = 0
    print("\u5de5\u4f4d  \u53c2\u8003\u70b9  \u5f85\u6d4b\u70b9  \u5750\u6807\u4e0d\u7b26  Z\u6700\u5927\u504f\u5dee(mm)  \u7ed3\u8bba")
    print("-" * 58)
    for st in stations:
        a = read_points(ref_dir / f"station_{st}_points.csv")
        b_path = test_dir / f"station_{st}_points.csv"
        if not b_path.exists():
            print(f"{st:>4}  {len(a):>6}      --  \u7f3a\u5c11\u8f93\u51fa\u6587\u4ef6")
            bad += 1
            continue
        b = read_points(b_path)
        n_diff = sum(1 for i in range(min(len(a), len(b)))
                     if a[i][:2] != b[i][:2])
        dz = max((abs(a[i][2] - b[i][2]) for i in range(min(len(a), len(b)))),
                 default=0.0)
        ok = (len(a) == len(b) and n_diff == 0 and dz <= Z_TOL)
        print(f"{st:>4}  {len(a):>6}  {len(b):>6}  {n_diff:>8}  {dz:>14.3f}  "
              f"{'\u4e00\u81f4' if ok else '\u4e0d\u4e00\u81f4'}")
        if not ok:
            bad += 1

    sa = read_summary(ref_dir / "summary.csv") if (ref_dir / "summary.csv").exists() else {}
    sb = read_summary(test_dir / "summary.csv") if (test_dir / "summary.csv").exists() else {}
    if sa and sb:
        diffs = []
        for st in sorted(sa):
            if st not in sb:
                diffs.append(f"station {st} \u7f3a\u5c11\u6c47\u603b\u884c")
                continue
            if abs(float(sa[st]["split_mm"]) - float(sb[st]["split_mm"])) > SPLIT_TOL:
                diffs.append(f"station {st} \u5206\u754c\u503c \u4e0d\u540c")
            for k in ("top_h_bars", "top_v_bars", "far_h_bars", "far_v_bars",
                      "accepted", "rejected_lower", "rejected_clutter"):
                if int(sa[st][k]) != int(sb[st][k]):
                    diffs.append(f"station {st} {k}: {sa[st][k]} vs {sb[st][k]}")
        if diffs:
            bad += 1
            print("\nsummary.csv \u5dee\u5f02\uff1a")
            for d in diffs:
                print("  " + d)
        else:
            print("\nsummary.csv\uff1a\u5b8c\u5168\u4e00\u81f4")

    print("\n" + ("\u2713 \u5168\u90e8\u4e00\u81f4\uff1a\u4e24\u7aef\u8f93\u51fa\u9010\u70b9\u76f8\u540c"
                  if bad == 0 else f"\u2717 \u6709 {bad} \u5904\u4e0d\u4e00\u81f4\uff0c\u9700\u6392\u67e5"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
