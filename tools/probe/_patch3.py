# -*- coding: utf-8 -*-
import io
p = r"D:\work\gygj\src\detect.py"
s = io.open(p, encoding="utf-8").read()

# 1) 新增：去掉骨架交叉节点
helper = '''
def remove_junctions(skel):
    """去掉骨架的分叉点，把交叉成网格的骨架拆成一段段单筋曲线。

    网格骨架在每个交叉处会有 T 形/十字形分叉，HoughLinesP 会把分叉处的对角
    短线当成斜线，污染筋条聚类（表现为同一条竖筋被拆成多簇、或冒出斜率反号的
    伪筋条）。先去掉分叉点即可消除。
    """
    img = (np.asarray(skel) > 0).astype(np.uint8)
    p = np.pad(img, 1)
    nbr = sum(p[1 + dy:1 + dy + img.shape[0], 1 + dx:1 + dx + img.shape[1]]
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return (img > 0) & (nbr < 3)


def _segment_ab(seg):'''
assert "\ndef _segment_ab(seg):" in s
s = s.replace("\ndef _segment_ab(seg):", helper, 1)

# 2) extract_bar_lines 中先拆骨架再取线段
old = """    skel = zhang_suen(closed, p["skel_max_iter"])

    segments = _hough_segments(skel, p)"""
new = """    skel = zhang_suen(closed, p["skel_max_iter"])
    branches = remove_junctions(skel)

    segments = _hough_segments(branches.astype(np.uint8) * 255, p)"""
assert old in s
s = s.replace(old, new, 1)

# 3) 重拟合也应该只在去分叉后的像素上做
old2 = """    ys, xs = np.nonzero(skel)
    lines = []
    if ys.size:"""
new2 = """    ys, xs = np.nonzero(branches)
    lines = []
    if ys.size:"""
assert old2 in s
s = s.replace(old2, new2, 1)

# 4) Hough 段长阈值提高
s = s.replace("hough_min_len=80,", "hough_min_len=150,", 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
