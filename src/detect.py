"""钢筋交叉点检测（尚未实现，属路线图第 3 步）。

计划算法：
  1. 在顶层掩膜上做形态学去毛刺 + 骨架化，得到单像素中心线；
  2. 用 HoughLinesP 提取线段，按角度聚成"横筋/竖筋"两组，
     每组做直线拟合（最小二乘/PCA）得到干净的筋轴线；
  3. 横竖两组直线两两求交，交点即候选交叉点；
  4. 剔除：交点邻域内若深度落在远层 -> 判为下层交叉点，丢弃；
     交点邻域深度明显浅于顶层（如 station_1 中 700mm 附近的非钢筋杂物）
     -> 判为虚假干扰点，丢弃；
  5. 输出保留点与两类被剔除点，供可视化区分显示。

接口约定：见 detect_intersections 的签名，返回值同时给出保留点和两类被拒点，
以便可视化模块按任务书要求区分显示。
"""
from __future__ import annotations


def detect_intersections(gray, depth_mm, **params):
    """检测顶层钢筋交叉点。

    参数
    ----
    gray      : uint8 单通道增强后的灰度图（仅用于纹理/边缘辅助判断）
    depth_mm  : float32 深度图，单位毫米，无效像素为 NaN，已与 gray 对齐
    **params  : 算法参数（阈值等），见模块 docstring

    返回
    ----
    dict，至少包含:
      "accepted": [(x, y, depth_mm), ...]   顶层交叉点
      "rejected_lower":  [...]              被判为下层而剔除的交叉点
      "rejected_clutter": [...]             被判为非钢筋杂物而剔除的点
    """
    raise NotImplementedError("路线图第 3 步实现，见 docs/交接文档.md")
