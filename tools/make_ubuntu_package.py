"""打包 Ubuntu 验证包：只带运行与验证需要的东西。

打包内容：app.py、app/（图形界面）、run.py、requirements.txt、src/、docs/、
tools/probe/*.py、tools/verify_outputs.py、任务书数据+代码/、
ref_win/（Windows 端产物，供逐点比对）。
排除：outputs/、outputs_app/、.git/、__pycache__、探针里的证据图与大图。

用法：python tools/make_ubuntu_package.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "gygj_ubuntu.zip"

INCLUDE_FILES = ["app.py", "run.py", "requirements.txt"]
INCLUDE_DIRS = ["src", "app", "docs"]
SKIP_DIRS = {"__pycache__", ".git", "outputs", "outputs_app", "preview", "media"}
SKIP_SUFFIX = {".pyc"}


def want(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    return path.suffix not in SKIP_SUFFIX


def add(zf, path: Path, arcname: str):
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and want(child.relative_to(ROOT)):
                zf.write(child, str(Path(arcname) / child.relative_to(path)))
    elif path.is_file():
        zf.write(path, arcname)


def main():
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in INCLUDE_FILES:
            add(zf, ROOT / f, f)
        for d in INCLUDE_DIRS:
            add(zf, ROOT / d, d)
        for py in (ROOT / "tools").glob("*.py"):
            zf.write(py, f"tools/{py.name}")
        for py in (ROOT / "tools" / "probe").glob("*.py"):
            zf.write(py, f"tools/probe/{py.name}")
        add(zf, ROOT / "\u4efb\u52a1\u4e66\u6570\u636e+\u4ee3\u7801", "\u4efb\u52a1\u4e66\u6570\u636e+\u4ee3\u7801")
        add(zf, ROOT / "ref_win", "ref_win")
        n = len(zf.namelist())
    size = OUT.stat().st_size / 1024 / 1024
    print(f"\u6253\u5305\u5b8c\u6210\uff1a{OUT}  ({n} \u4e2a\u6587\u4ef6, {size:.1f} MB)")


if __name__ == "__main__":
    main()
