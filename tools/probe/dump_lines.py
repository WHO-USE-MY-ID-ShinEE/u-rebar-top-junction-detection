"""打印某站检测到的全部筋线参数，用于人工核对条数与轴线。

用法：python tools/probe/dump_lines.py 站点号
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import detect, imgio, pipeline  # noqa: E402


def main(argv):
    st = int(argv[0])
    gray, depth_mm, _ = imgio.load_station(pipeline.DEFAULT_DATA_DIR, st)
    res = detect.detect_rebar_intersections(depth_mm, gray)
    print(f"station_{st}  深度分界={res['split_mm']:.1f}mm")
    for tag, key in (("顶层横筋", "h_lines"), ("顶层竖筋", "v_lines"),
                     ("下层横筋", "far_h_lines"), ("下层竖筋", "far_v_lines")):
        lines = res[key]
        print(f"\n{tag}：{len(lines)} 条")
        for l in sorted(lines, key=lambda z: z["ab"][1]):
            a, b = l["ab"]
            ex = l["extent"]
            print(f"  a={a:+.4f} b={b:7.1f} span={l['span']:6.1f} "
                  f"cov={l['coverage']:.2f} n={l['n_support']:5d} "
                  f"({ex[0][0]:6.1f},{ex[0][1]:6.1f})->({ex[1][0]:6.1f},{ex[1][1]:6.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
