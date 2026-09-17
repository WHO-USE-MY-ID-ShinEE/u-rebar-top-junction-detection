"""课题四 顶层钢筋交叉点检测系统 —— 图形界面（Tkinter）。

布局：
    顶部  工具栏：打开图像对 / 打开数据集 / 分析 / 批量分析 / 导出 / 帮助
    左栏  数据源选择（数据集工位或任意图像对）+ 样本缩略图
    中栏  图像画布：滚轮缩放、左键拖拽平移、左键单击查询交叉点
    右栏  检测参数、显示选项、统计与完备性审计
    底部  状态栏：当前样本、处理耗时、保留点数

界面只做事件绑定与显示，全部计算走 app.runner，与命令行流水线、无界面自检
（python app.py --selftest）共用同一条代码路径，避免"界面一套、脚本另一套"。

对应任务书的四项完成标准：
    * 交叉点检测结果在图上标注，坐标可查、可导出 csv / json；
    * 视图可在「原始灰度图 / 增强灰度图 / 深度伪彩图 / 检测结果」之间切换；
    * 被过滤掉的下层交叉点、虚假干扰点分色显示，且可分别开关；
    * 批量跑多组样本，给出漏检数、误检数、完备率（准确率），参数可在线调整。
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src import imgio, pipeline

from . import render, runner

VIEWS = render.VIEWS
OPTION_LABELS = {
    "bars": "显示筋条轴线",
    "skeleton": "显示顶层骨架",
    "lower": "显示被剔除的下层交叉点",
    "clutter": "显示被剔除的虚假干扰点",
    "labels": "给保留点编号",
}
HELP_TEXT = """课题四 · U 型截面多层钢筋笼 顶层交叉点深度视觉检测

【系统用途】
    用同一场景配对的「深度图 + 灰度图」找出钢筋笼顶层骨架的钢筋交叉点，
    并把坐标输出给自动捆扎机器人；同时排除下层钢筋与现场非钢筋杂物造成的
    虚假交叉点。

【输入 / 输出】
    输入：深度图（.tif，单位米，无效像素为 NaN）+ 灰度图（.png）
          深度图与灰度图互为转置时程序自动逆时针旋转 90° 对齐
    输出：交叉点坐标 csv（point_id,x_px,y_px,z_mm）
          交叉点坐标 json（points: [[x, y], ...]）
          被剔除点 json（lower: 下层点，clutter: 虚假干扰点）
          一页式结果图 png（结果 / 分层 / 掩膜骨架 / 统计四格）

【算法流程】
    1. 有效像素掩膜：深度有限且大于 0，其余视为无回波背景；
    2. 深度分层：在深度直方图上找最宽空段自动定界，把近层判为顶层，
       也可以手工指定分界值；
    3. 顶层掩膜 -> 形态学闭运算 -> Zhang-Suen 骨架化 -> 去掉分叉点
       -> HoughLinesP 取线段 -> 按"族"参数化聚类（横筋 y=ax+b、竖筋 x=ay+b）
       -> 精拟合得到每条筋的轴线、跨度、覆盖度；
    4. 顶层横筋 x 顶层竖筋两两求交得到候选点，逐点做三条判据：
       a. 深度判据：交点处深度小于分界值，否则判为下层点；
       b. 支撑连续性：沿两条筋的方向局部都有连续掩膜，断口/勉强连成的线判为干扰；
       c. 实心块判据：交点四角被整片填满说明压在实心杂物上，判为干扰；
    5. 邻域去重后输出保留点，并做一次几何完备性自审计
       （把几何上应当存在的交点全部列出，与保留点比对，得到漏检/误检）。

【界面颜色约定】
    红点 = 保留的顶层交叉点      蓝点 = 被剔除的下层交叉点
    橙点 = 被剔除的虚假干扰点    品红线 = 横筋轴线   青线 = 竖筋轴线

【操作步骤】
    1. 左栏选择数据源：默认读数据集目录下的 10 个工位，也可以"打开图像对"；
    2. 在样本列表里选中一条，中栏立刻显示原始/深度图像预览；
    3. 按"分析"（F5）开始检测，结果显示在中栏与右栏；
    4. 中栏切换视图、滚轮缩放、左键单击任意交叉点查看坐标与深度；
    5. "导出结果"把当前样本的坐标文件与结果图存到指定目录；
    6. "批量分析"跑完所有工位，给出每站的漏检数、误检数、完备率，
       并可把汇总表导出成 csv 供报告使用。

【常见问题】
    * Ubuntu 上界面中文变方块：sudo apt install fonts-noto-cjk
    * 结果图中文变方块：同上（结果图用系统 CJK 字体渲染中文标注）
    * 提示找不到 tkinter：sudo apt install python3-tk
    * 提示没有找到工位数据：确认数据目录已正确解压（目录名含中文，
      建议用 python3 -c "import zipfile;zipfile.ZipFile('包名').extractall()"）
"""


class CanvasView(ttk.Frame):
    """可缩放、可平移的图像画布；左键单击回调图像像素坐标。"""

    def __init__(self, master, on_click=None):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.scale = 1.0
        self.src = None
        self.photo = None
        self.marker = None
        self.on_click = on_click
        self._drag = None

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<MouseWheel>", self._wheel)                   # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.25, e))   # Linux 滚轮上
        self.canvas.bind("<Button-5>", lambda e: self._zoom(0.8, e))    # Linux 滚轮下

    # ---------- 外部接口 ----------
    def set_image(self, bgr):
        self.src = bgr
        self.marker = None
        self._show()

    def fit(self):
        if self.src is None:
            return
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        h, w = self.src.shape[:2]
        self.scale = min(cw / w, ch / h)
        self._show()

    def actual_size(self):
        self.scale = 1.0
        self._show()

    def mark(self, pt):
        """在图像坐标 pt 处画一个定位圈（用于点查询）。"""
        self.marker = pt
        self._show()

    # ---------- 内部 ----------
    def _show(self):
        if self.src is None:
            return
        h, w = self.src.shape[:2]
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_NEAREST
        disp = cv2.resize(self.src, (nw, nh), interpolation=interp)
        if self.marker is not None:
            cx = int(self.marker[0] * self.scale)
            cy = int(self.marker[1] * self.scale)
            cv2.circle(disp, (cx, cy), 18, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(disp, (cx, cy), 20, (0, 0, 0), 1, cv2.LINE_AA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, nw, nh))

    def _wheel(self, event):
        if event.delta:
            self._zoom(1.25 if event.delta > 0 else 0.8, event)

    def _zoom(self, factor, event):
        if self.src is None:
            return
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        ix, iy = cx / self.scale, cy / self.scale
        self.scale = float(min(12.0, max(0.05, self.scale * factor)))
        self._show()
        h, w = self.src.shape[:2]
        self.canvas.xview_moveto(max(0.0, min(1.0, (ix * self.scale - event.x) / (w * self.scale))))
        self.canvas.yview_moveto(max(0.0, min(1.0, (iy * self.scale - event.y) / (h * self.scale))))

    def _press(self, event):
        self._drag = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def _motion(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _release(self, event):
        if self._drag is None:
            return
        moved = abs(event.x - self._drag[0]) + abs(event.y - self._drag[1])
        self._drag = None
        if moved <= 4 and self.on_click is not None:
            self.on_click(self.canvas.canvasx(event.x) / self.scale,
                          self.canvas.canvasy(event.y) / self.scale)


class BatchWindow(tk.Toplevel):
    """批量检测窗口：进度条 + 每站统计表。"""

    COLUMNS = (
        ("name", "样本", 90), ("split_mm", "深度分界 mm", 100),
        ("h_bars", "顶层横筋", 80), ("v_bars", "顶层竖筋", 80),
        ("accepted", "保留点", 70), ("lower", "下层点", 70), ("clutter", "干扰点", 70),
        ("expected", "期望点", 70), ("matched", "匹配点", 70),
        ("misses", "漏检", 60), ("extra", "误检", 60),
        ("completeness", "完备率 %", 80), ("elapsed", "耗时 s", 70),
    )

    def __init__(self, master, count):
        super().__init__(master)
        self.title("批量检测结果")
        self.geometry("1120x520")
        self.transient(master)
        self.stop_flag = False

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.bar = ttk.Progressbar(top, maximum=max(count, 1), mode="determinate")
        self.bar.pack(side="left", fill="x", expand=True)
        self.progress_label = ttk.Label(top, text="准备开始…", width=42, anchor="w")
        self.progress_label.pack(side="left", padx=8)
        ttk.Button(top, text="停止", command=self._stop).pack(side="right")
        ttk.Button(top, text="导出汇总表", command=master.export_summary).pack(side="right", padx=6)

        body = ttk.Frame(self, padding=(8, 0))
        body.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(body, columns=[c[0] for c in self.COLUMNS],
                                 show="headings", height=14)
        for key, title, width in self.COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="center")
        vbar = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        self.summary_label = ttk.Label(self, text="", padding=8, anchor="w")
        self.summary_label.pack(fill="x")

    def _stop(self):
        self.stop_flag = True
        self.progress_label.config(text="正在停止…请等待当前工位结束")

    def progress(self, text, done, total):
        """由后台线程通过队列间接调用（不能在子线程里直接碰界面）。"""
        self.progress_label.config(text=text)
        self.bar.config(value=done)

    def fill(self, rows):
        keys = [c[0] for c in self.COLUMNS]
        for row in rows:
            self.tree.insert("", "end", values=[row.get(k, "") for k in keys])
        if rows:
            avg = sum(r["completeness"] for r in rows) / len(rows)
            total_miss = sum(r["misses"] - r["extra"] for r in rows)
            self.summary_label.config(
                text=f"共 {len(rows)} 站：保留点 {sum(r['accepted'] for r in rows)} 个，"
                     f"被剔除的下层点 {sum(r['lower'] for r in rows)} 个，"
                     f"被剔除的干扰点 {sum(r['clutter'] for r in rows)} 个；"
                     f"平均完备率 {avg:.1f}%（完整表可导出为 csv）")
            self.progress_label.config(text="批量检测完成")


class App(tk.Tk):
    """主窗口。"""

    def __init__(self, data_dir=None):
        super().__init__()
        self.title("课题四 · U 型钢筋笼顶层交叉点检测系统")
        self.geometry("1520x920")
        self.minsize(1120, 720)
        _install_cjk_font(self)

        self.data_dir = Path(data_dir) if data_dir else Path(pipeline.DEFAULT_DATA_DIR)
        self.out_dir = Path("outputs_app")
        self.stations = []
        self.pairs = []
        self.state = None
        self.rows = []
        self.states = []
        self.batch_win = None
        self._busy = False
        self._queue = queue.Queue()

        self.mode = tk.StringVar(value="station")
        self.view = tk.StringVar(value=VIEWS[3])
        self.option_vars = {k: tk.BooleanVar(value=v)
                            for k, v in render.DEFAULT_OPTIONS.items()}
        self.param_vars = {}

        self._build()
        self.refresh_stations()
        if self.stations:
            self.listbox.selection_set(0)
            self.load_selected()
        self.after(80, self._drain)

    # ---------------- 界面搭建 ----------------
    def _build(self):
        self._build_toolbar()

        body = ttk.Frame(self, padding=(8, 0, 8, 4))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_left(body)
        self.viewer = CanvasView(body, on_click=self.query_point)
        self.viewer.grid(row=0, column=1, sticky="nsew", padx=6)
        self._build_right(body)
        self._build_status()

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(8, 8, 8, 4))
        bar.pack(fill="x")
        self._buttons = []

        def button(text, command, width=None):
            b = ttk.Button(bar, text=text, command=command, width=width)
            b.pack(side="left", padx=3)
            self._buttons.append(b)
            return b

        button("打开图像对…", self.open_pair)
        button("打开数据集…", self.choose_data_dir)
        button("分析 (F5)", self.analyze_current)
        button("批量分析", self.batch_analyze)
        button("导出结果", self.export_current)
        button("导出汇总表", self.export_summary)
        button("帮助", self.show_help)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(bar, text="视图：").pack(side="left")
        combo = ttk.Combobox(bar, textvariable=self.view, values=list(VIEWS),
                             state="readonly", width=12)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        button("适应窗口", lambda: self.viewer.fit())
        button("1:1", lambda: self.viewer.actual_size())
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        button("上一张", lambda: self.step(-1), width=7)
        button("下一张", lambda: self.step(1), width=7)

        self.bind("<F5>", lambda e: self.analyze_current())
        self.bind("<Control-o>", lambda e: self.open_pair())

    def _build_left(self, parent):
        frame = ttk.Frame(parent, width=250)
        frame.grid(row=0, column=0, sticky="ns")
        frame.grid_propagate(False)

        box = ttk.LabelFrame(frame, text="数据源", padding=6)
        box.pack(fill="x")
        self.data_label = ttk.Label(box, text="", wraplength=220, foreground="#333")
        self.data_label.pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Radiobutton(row, text="数据集工位", value="station", variable=self.mode,
                        command=self._fill_list).pack(anchor="w")
        ttk.Radiobutton(row, text="已打开的图像对", value="pair", variable=self.mode,
                        command=self._fill_list).pack(anchor="w")

        box2 = ttk.LabelFrame(frame, text="样本", padding=6)
        box2.pack(fill="both", expand=True, pady=6)
        self.listbox = tk.Listbox(box2, exportselection=False, activestyle="dotbox")
        sbar = ttk.Scrollbar(box2, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sbar.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sbar.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.load_selected())
        self.listbox.bind("<Double-Button-1>", lambda e: self.analyze_current())

        box3 = ttk.LabelFrame(frame, text="缩略图", padding=6)
        box3.pack(fill="x")
        self.thumb = tk.Canvas(box3, width=214, height=150, bg="#1e1e1e",
                               highlightthickness=0)
        self.thumb.pack()
        self._thumb_photo = None

    def _build_right(self, parent):
        frame = ttk.Frame(parent, width=290)
        frame.grid(row=0, column=2, sticky="ns")
        frame.grid_propagate(False)

        box = ttk.LabelFrame(frame, text="检测参数", padding=6)
        box.pack(fill="x")
        for i, (key, label, hint) in enumerate(runner.TUNABLE + runner.TUNABLE_EXTRA):
            ttk.Label(box, text=label).grid(row=i * 2, column=0, sticky="w", pady=(4, 0))
            var = tk.StringVar(value=runner.default_text(key))
            self.param_vars[key] = var
            ttk.Entry(box, textvariable=var, width=10).grid(row=i * 2, column=1,
                                                            sticky="e", pady=(4, 0))
            ttk.Label(box, text=hint, foreground="#777", wraplength=260,
                      justify="left").grid(row=i * 2 + 1, column=0, columnspan=2, sticky="w")
        box.columnconfigure(0, weight=1)
        ttk.Button(box, text="恢复默认参数", command=self.reset_params).grid(
            row=99, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        box2 = ttk.LabelFrame(frame, text="显示选项", padding=6)
        box2.pack(fill="x", pady=6)
        for key, label in OPTION_LABELS.items():
            ttk.Checkbutton(box2, text=label, variable=self.option_vars[key],
                            command=self.redraw).pack(anchor="w")

        box3 = ttk.LabelFrame(frame, text="统计与完备性审计", padding=6)
        box3.pack(fill="both", expand=True)
        self.stats_text = tk.Text(box3, width=34, height=26, wrap="word",
                                  relief="flat", background="#f5f5f5")
        self.stats_text.pack(fill="both", expand=True)
        self.stats_text.configure(state="disabled")

    def _build_status(self):
        bar = ttk.Frame(self, padding=(8, 2, 8, 6))
        bar.pack(fill="x")
        self.status = ttk.Label(bar, text="就绪", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self.pick_label = ttk.Label(bar, text="左键单击交叉点可查看坐标", anchor="e")
        self.pick_label.pack(side="right")


    # ---------------- 数据源 ----------------
    def refresh_stations(self):
        self.stations = imgio.list_stations(self.data_dir) if self.data_dir.exists() else []
        self.data_label.config(text=str(self.data_dir))
        if self.mode.get() == "station":
            self._fill_list()

    def _fill_list(self):
        self.listbox.delete(0, tk.END)
        if self.mode.get() == "station":
            for n in self.stations:
                self.listbox.insert(tk.END, f"station_{n}")
        else:
            for p in self.pairs:
                self.listbox.insert(tk.END, p["name"])

    def choose_data_dir(self):
        path = filedialog.askdirectory(title="选择数据集目录（含 station_N 图像）",
                                       initialdir=str(self.data_dir))
        if not path:
            return
        self.data_dir = Path(path)
        self.pairs = []
        self.mode.set("station")
        self.refresh_stations()
        if self.stations:
            self.listbox.selection_set(0)
            self.load_selected()
        else:
            messagebox.showwarning(
                "没有找到工位数据",
                f"{path} 下没有 station_N 图像。\n"
                "请确认数据目录已正确解压（解压工具可能改坏中文目录名，\n"
                "建议用 python3 -c \"import zipfile;zipfile.ZipFile('包名').extractall()\"）。")

    def open_pair(self):
        depth = filedialog.askopenfilename(
            title="选择深度图",
            filetypes=[("图像文件", "*.tif *.tiff *.png *.jpg"), ("全部文件", "*.*")])
        if not depth:
            return
        gray = filedialog.askopenfilename(
            title="选择配套灰度图（取消则只用深度图）",
            filetypes=[("图像文件", "*.png *.jpg *.tif *.tiff"), ("全部文件", "*.*")])
        self.pairs.append({"name": runner.name_for_paths(depth), "depth": depth,
                           "gray": gray or None})
        self.mode.set("pair")
        self._fill_list()
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(len(self.pairs) - 1)
        self.load_selected()

    def _selected_source(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        if self.mode.get() == "station":
            if sel[0] >= len(self.stations):
                return None
            return {"kind": "station", "data_dir": str(self.data_dir),
                    "station": self.stations[sel[0]]}
        if sel[0] >= len(self.pairs):
            return None
        p = self.pairs[sel[0]]
        return {"kind": "pair", "depth": p["depth"], "gray": p["gray"]}

    # ---------------- 加载与检测 ----------------
    def load_selected(self):
        if self._busy:
            return
        src = self._selected_source()
        if src is None:
            return
        try:
            _, clip = self._params()
        except ValueError as exc:
            messagebox.showerror("参数有误", str(exc))
            return
        label = self.listbox.get(self.listbox.curselection()[0])

        def work():
            if src["kind"] == "pair":
                return runner.load(depth_path=src["depth"], gray_path=src["gray"],
                                   clahe_clip=clip)
            return runner.load(data_dir=src["data_dir"], station=src["station"],
                               clahe_clip=clip)

        self._need_fit = True
        self._submit(work, self._on_loaded, f"正在读取 {label} …")

    def _on_loaded(self, st):
        self.state = st
        self.redraw()
        self._update_stats()
        h, w = st["depth_mm"].shape[:2]
        self.set_status(f"{st['name']}：{w} x {h} 像素，已载入。按「分析」（F5）开始检测。")

    def analyze_current(self):
        if self._busy:
            return
        if self.state is None:
            self.load_selected()
            return
        try:
            params, clip = self._params()
        except ValueError as exc:
            messagebox.showerror("参数有误", str(exc))
            return
        src = dict(self.state["source"])

        def work():
            if src["kind"] == "pair":
                return runner.analyze(depth_path=src["depth"], gray_path=src["gray"],
                                      params=params, clahe_clip=clip)
            return runner.analyze(data_dir=src["data_dir"], station=src["station"],
                                  params=params, clahe_clip=clip)

        self._submit(work, self._on_analyzed, f"正在检测 {self.state['name']} …")

    def _on_analyzed(self, st):
        self.state = st
        self.redraw()
        self._update_stats()
        s = st["stats"]
        self.set_status(
            f"{st['name']}：保留 {s['accepted']} 个顶层交叉点，"
            f"剔除下层点 {s['lower']} 个、干扰点 {s['clutter']} 个，"
            f"耗时 {st['elapsed']:.2f} 秒。")

    def _params(self):
        """读界面参数，返回 (检测参数, CLAHE 对比度或 None)。"""
        texts = {k: self.param_vars[k].get() for k, _, _ in runner.TUNABLE}
        params = runner.parse_params(texts)
        clip = runner.parse_params(
            {"clahe_clip": self.param_vars["clahe_clip"].get().strip()}).get("clahe_clip")
        return params, clip

    def reset_params(self):
        for key, var in self.param_vars.items():
            var.set(runner.default_text(key))
        self.set_status("参数已恢复默认值。")

    # ---------------- 显示 ----------------
    def _options(self):
        return {k: v.get() for k, v in self.option_vars.items()}

    def redraw(self):
        st = self.state
        if st is None:
            return
        img = render.view_image(self.view.get(), st, self._options())
        self.viewer.set_image(img)
        if getattr(self, "_need_fit", False):
            self.viewer.fit()
            self._need_fit = False
        self._draw_thumbnail(img)

    def _draw_thumbnail(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(214.0 / w, 150.0 / h)
        small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        self._thumb_photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.thumb.delete("all")
        self.thumb.create_image(107, 75, image=self._thumb_photo)

    def query_point(self, ix, iy):
        st = self.state
        if st is None or st["res"] is None:
            self.pick_label.config(text=f"像素 ({ix:.0f}, {iy:.0f})（尚未检测）")
            return
        groups = (("顶层交叉点", st["res"]["accepted"], "P"),
                  ("已剔除的下层点", st["res"]["rejected_lower"], "L"),
                  ("已剔除的干扰点", st["res"]["rejected_clutter"], "X"))
        tol = max(14.0, 20.0 / max(self.viewer.scale, 1e-6))
        best = None
        for name, pts, tag in groups:
            for i, p in enumerate(pts, 1):
                dist = float(np.hypot(p[0] - ix, p[1] - iy))
                if dist <= tol and (best is None or dist < best[0]):
                    best = (dist, name, tag, i, p)
        if best is None:
            self.viewer.mark((ix, iy))
            self.pick_label.config(text=f"像素 ({ix:.0f}, {iy:.0f}) 附近没有交叉点")
            return
        _, name, tag, i, p = best
        depth = f"，深度 {p[2]:.1f} mm" if len(p) > 2 else ""
        self.viewer.mark((p[0], p[1]))
        self.pick_label.config(
            text=f"{name} {tag}{i:03d}：({int(p[0])}, {int(p[1])}){depth}")

    def _update_stats(self):
        st = self.state
        if st is None:
            text = "尚未选择样本。"
        elif st["res"] is None:
            text = (f"样本：{st['name']}\n\n尚未检测。\n"
                    "按「分析」（F5）开始检测，\n或双击左侧样本。")
        else:
            text = self._stats_text(st)
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", text)
        self.stats_text.configure(state="disabled")

    def _stats_text(self, st):
        s, a = st["stats"], st["audit"]
        h, w = st["depth_mm"].shape[:2]
        lines = [
            f"样本：{st['name']}",
            f"尺寸：{w} x {h} 像素",
            f"深度分界：{st['res']['split_mm']:.1f} mm",
            "",
            f"顶层横筋 / 竖筋：{s['h_bars']} / {s['v_bars']} 条",
            f"下层横筋 / 竖筋：{s['far_h_bars']} / {s['far_v_bars']} 条",
            "",
            f"保留的顶层交叉点：{s['accepted']} 个",
            f"剔除的下层交叉点：{s['lower']} 个",
            f"剔除的虚假干扰点：{s['clutter']} 个",
            "",
            "—— 完备性审计（无需人工真值）——",
            f"几何期望交叉点：{a['expected']} 个",
            f"其中被保留：{a['matched']} 个",
            f"完备率：{a['completeness'] * 100:.1f}%",
            f"误检（多余点）：{len(a['extra'])} 个",
            f"漏检（原因不明的未保留点）：{a['lost']} 个",
        ]
        if a["reasons"]:
            lines += ["", "未保留点的原因："]
            lines += [f"  · {k}：{v} 个" for k, v in a["reasons"].items()]
        lines += ["", f"耗时：{st['elapsed']:.2f} 秒"]
        return "\n".join(lines)

    # ---------------- 导出 ----------------
    def export_current(self):
        if self.state is None or self.state["res"] is None:
            messagebox.showinfo("提示", "请先分析一个样本，再导出结果。")
            return
        out = filedialog.askdirectory(title="选择导出目录", initialdir=str(self.out_dir))
        if not out:
            return
        files = runner.export(self.state, out)
        self.out_dir = Path(out)
        self.set_status(f"已导出到 {out}")
        messagebox.showinfo("导出完成", "已生成：\n" + "\n".join(str(v) for v in files.values()))

    def export_summary(self):
        if not self.rows:
            messagebox.showinfo("提示", "还没有批量结果，请先做一次「批量分析」。")
            return
        path = filedialog.asksaveasfilename(title="保存汇总表", defaultextension=".csv",
                                            initialfile="app_summary.csv",
                                            initialdir=str(self.out_dir))
        if not path:
            return
        p = runner.write_summary(self.rows, Path(path).parent, filename=Path(path).name)
        self.set_status(f"汇总表已保存：{p}")
        messagebox.showinfo("导出完成", f"汇总表已保存：\n{p}")

    # ---------------- 批量 ----------------
    def batch_analyze(self):
        if self._busy:
            return
        if not self.stations:
            messagebox.showinfo("提示", "当前数据源里没有工位数据，请先「打开数据集」。")
            return
        try:
            params, clip = self._params()
        except ValueError as exc:
            messagebox.showerror("参数有误", str(exc))
            return
        stations = list(self.stations)
        win = BatchWindow(self, len(stations))
        self.batch_win = win

        def work():
            return runner.batch(
                stations, str(self.data_dir), params=params, clahe_clip=clip,
                on_progress=lambda *a: self._queue.put(("progress", None, a)),
                should_stop=lambda: win.stop_flag)

        def done(result):
            rows, states = result
            self.rows, self.states = rows, states
            win.fill(rows)
            self.set_status(f"批量检测完成：{len(rows)} 站。可点「导出汇总表」保存统计表。")

        self._submit(work, done, "批量检测中…")

    # ---------------- 其它 ----------------
    def step(self, delta):
        size = self.listbox.size()
        if not size or self._busy:
            return
        sel = self.listbox.curselection()
        idx = (sel[0] if sel else -1) + delta
        idx = max(0, min(size - 1, idx))
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self.load_selected()

    def show_help(self):
        win = tk.Toplevel(self)
        win.title("使用说明与算法说明")
        win.geometry("760x620")
        win.transient(self)
        text = tk.Text(win, wrap="word", padx=12, pady=10)
        sbar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=sbar.set)
        text.pack(side="left", fill="both", expand=True)
        sbar.pack(side="right", fill="y")
        text.insert("1.0", HELP_TEXT)
        text.configure(state="disabled")

    def set_status(self, text):
        self.status.config(text=text)

    def _set_busy(self, flag):
        for b in self._buttons:
            b.config(state="disabled" if flag else "normal")

    def _submit(self, work, done, text):
        """把耗时计算放到后台线程，结果通过队列回到界面线程。"""
        if self._busy:
            return
        self._busy = True
        self._set_busy(True)
        self.set_status(text)

        def run():
            try:
                self._queue.put(("ok", done, work()))
            except Exception as exc:                       # noqa: BLE001
                self._queue.put(("err", done, exc))

        threading.Thread(target=run, daemon=True).start()

    def _drain(self):
        try:
            while True:
                kind, done, payload = self._queue.get_nowait()
                if kind == "progress":
                    if self.batch_win is not None:
                        self.batch_win.progress(*payload)
                    continue
                self._busy = False
                self._set_busy(False)
                if kind == "ok":
                    done(payload)
                else:
                    self.set_status(f"出错：{payload}")
                    messagebox.showerror("出错了", str(payload))
        except queue.Empty:
            pass
        self.after(80, self._drain)


def _install_cjk_font(root):
    """挑一个装了的中文字体当界面字体（Ubuntu 默认字体没有中文字形）。"""
    from tkinter import font as tkfont

    try:
        names = set(tkfont.families(root))
    except tk.TclError:
        return None
    for cand in ("Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                 "Source Han Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                 "SimHei", "PingFang SC"):
        if cand in names:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                         "TkHeadingFont"):
                try:
                    tkfont.nametofont(name).configure(family=cand, size=10)
                except tk.TclError:
                    pass
            return cand
    return None


def main(data_dir=None):
    """启动图形界面。"""
    App(data_dir=data_dir).mainloop()
