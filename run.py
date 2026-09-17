"""命令行入口。

用法:
    python run.py --all                   # 跑全部站点预览
    python run.py --stations 1 6          # 跑指定站点
    python run.py --split-mm 840          # 手动指定分层阈值
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import imgio, pipeline  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="处理数据目录中的全部站点")
    ap.add_argument("--data-dir", default=str(pipeline.DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=str(pipeline.DEFAULT_OUT_DIR))
    ap.add_argument("--stage", default="preview", choices=["preview"])
    ap.add_argument("--split-mm", type=float, default=None,
                    help="顶层/下层深度分界；不给则按深度空段自动定界")
    args = ap.parse_args(argv)

    stations = args.stations or imgio.list_stations(args.data_dir)
    kwargs = {"data_dir": args.data_dir, "out_dir": args.out_dir}
    if args.split_mm is not None:
        kwargs["split_mm"] = args.split_mm

    print(f"stage={args.stage}  stations={stations}")
    for r in pipeline.run_stations(stations, stage=args.stage, **kwargs):
        s = r["stats"]
        n, f = s["near"], s["far"]
        auto = s["auto_split_mm"]
        print(f"  station {r['station']:2d}: split={s['split_mm']:6.1f}mm "
              f"(auto={'--' if auto is None else f'{auto:.1f}'}) "
              f"空段={s['gap_width_mm']:5.1f}mm  "
              f"near={n['px']:7d}({n['median_mm']:6.1f}mm) "
              f"far={f['px']:7d}({f['median_mm']:6.1f}mm) "
              f"可分离={s['separable']}")
        if not s["separable"]:
            print(f"    !! 该站点固定阈值不可靠，需要顶层平面拟合 -> {r['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
