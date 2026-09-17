"""命令行入口。

用法:
    python run.py --all                        # 全部站点，分层预览
    python run.py --all --stage detect         # 检测交叉点并出标注图
    python run.py --all --stage report         # 出交付用结果图 + 坐标文件 + 汇总表
    python run.py --stations 1 --split-mm 840  # 手动指定分层阈值（调试用）
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
    ap.add_argument("--stage", default="preview",
                    choices=["preview", "detect", "report"])
    ap.add_argument("--split-mm", type=float, default=None,
                    help="顶层/下层深度分界；不给则按深度空段自动定界")
    args = ap.parse_args(argv)

    stations = args.stations or imgio.list_stations(args.data_dir)

    if args.stage == "report":
        print(f"stage=report  stations={stations}")
        rows = pipeline.run_all_reports(stations, data_dir=args.data_dir,
                                        out_dir=args.out_dir)
        print(f"\n{'站':>3} {'保留':>5} {'下层':>5} {'干扰':>5}")
        for r in rows:
            print(f"{r['station']:>3} {r['accepted']:>5} "
                  f"{r['rejected_lower']:>5} {r['rejected_clutter']:>5}")
        return 0

    kwargs = {"data_dir": args.data_dir, "out_dir": args.out_dir}
    if args.split_mm is not None:
        kwargs["split_mm"] = args.split_mm

    print(f"stage={args.stage}  stations={stations}")
    for r in pipeline.run_stations(stations, stage=args.stage, **kwargs):
        print(f"  station {r['station']:2d} -> {r['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
