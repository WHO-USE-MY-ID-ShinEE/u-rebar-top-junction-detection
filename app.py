"""课题四 顶层钢筋交叉点检测系统 —— 应用入口。

用法：
    python app.py                     # 打开图形界面（默认读任务书数据集目录）
    python app.py --data-dir 目录      # 指定数据集目录后打开界面
    python app.py --selftest          # 无界面自检（不需要图形环境）

Ubuntu 22.04 首次运行前先装依赖：
    sudo apt install python3-tk python3-venv fonts-noto-cjk
    pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="课题四 钢筋交叉点检测系统")
    parser.add_argument("--data-dir", default=None,
                        help="数据集目录（含 station_N 的深度图与灰度图）")
    parser.add_argument("--selftest", action="store_true",
                        help="只做无界面自检，不打开图形界面")
    args = parser.parse_args(argv)

    if args.selftest:
        from app import selftest
        return selftest.main()

    try:
        from app import gui
        import tkinter
    except ImportError as exc:
        print("无法启动图形界面：", exc)
        print("Ubuntu 下请先安装 Tkinter：sudo apt install python3-tk")
        return 2

    try:
        gui.main(data_dir=args.data_dir)
    except tkinter.TclError as exc:
        print("没有可用的图形显示，无法打开界面：", exc)
        print("请在桌面环境里运行；只做验证可以改用：python app.py --selftest")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
