# -*- coding: utf-8 -*-
"""六种基材的正式模型：照度—时间损伤 f(E,t) 与光谱响应率 P(λ)。

数据来源：《基材公式补充.docx》（正式基材公式），替代《公式集合_初版》中
六种基材的早期过渡版本。口径见《需求确认记录.md》决议 R10–R16。

实现约定：
- **符号**：文档中 E 表示照度（lx），与本项目其余部分的 I 为同一物理量；
  e 为科学计数法（如 ``3.595e+07`` = 3.595×10⁷）。
- **照度—时间公式** 对应原"照度—时间损伤模型" f(E,t)；
  **光谱响应率公式** 对应原"波长响应函数" P(λ)。
- **pH 为各基材的固定属性**（公式在该 pH 下成立），不再作为计算变量或界面输入，
  仅作界面提示（见 PH_BY_SUBSTRATE）。
- **f 负值时按 0 处理**（决议 R11）：避免负损伤进入优化目标。
- **P(λ) 不做裁剪**（决议 R8b / R14）：负值直接参与 K 的梯形积分。

# =============================================================
# SUBSTRATE_MODEL_SECTION_START
# 集中隔离区：后续若有基材模型更新，仅替换本区内对应函数，
# 不修改 UI、damage_chain.py、optimizer.py。
# =============================================================
"""
from __future__ import annotations

import numpy as np

# 各基材的固定 pH 属性（仅用于界面提示；不参与任何计算）
PH_BY_SUBSTRATE: dict[str, float] = {
    "newsprint": 5.0,
    "bamboo_paper": 7.4,
    "xuan_paper": 8.5,
    "hemp_paper": 7.9,
    "mulberry_paper": 8.3,
    "silk": 5.9,
}


def _nonneg(value: float) -> float:
    """f(E,t) < 0 时按 0 处理（决议 R11）。"""
    return max(float(value), 0.0)


# ---------------------------------------------------------------------------
# 照度—时间损伤 f(E, t)   （E 为照度 lx，t 为时间 h）
# ---------------------------------------------------------------------------
def quantity_newsprint(E, t):  # 新闻纸 pH 5.0
    return _nonneg(
        -0.3421 + 0.01458 * E + 0.007503 * t - 0.0001127 * E**2
        + 6.322e-05 * E * t - 7.19e-05 * t**2
        + 2.452e-07 * E**3 + 1.762e-08 * E**2 * t
        - 1.627e-07 * E * t**2 + 1.523e-07 * t**3
    )


def quantity_bamboo_paper(E, t):  # 竹纸 pH 7.4
    return _nonneg(
        -0.3243 + 0.01883 * E + 0.04584 * t - 0.0001011 * E**2
        + 4.822e-05 * E * t - 0.0003088 * t**2
        + 1.779e-07 * E**3 + 9.369e-08 * E**2 * t
        - 1.632e-07 * E * t**2 + 5.324e-07 * t**3
    )


def quantity_xuan_paper(E, t):  # 宣纸 pH 8.5
    return _nonneg(
        -0.7135 + 0.01648 * E + 0.01581 * t + (-8.398e-05) * E**2
        + 2.313e-05 * E * t + (-5.914e-05) * t**2
        + 1.442e-07 * E**3 + 6.651e-08 * E**2 * t
        + (-1.188e-07) * E * t**2 + 8.27e-08 * t**3
    )


def quantity_hemp_paper(E, t):  # 麻纸 pH 7.9
    return _nonneg(
        3.147 - 0.06264 * E + 0.03753 * t + 0.0003611 * E**2
        + 5.717e-05 * E * t - 0.0001808 * t**2
        + (-5.994e-07) * E**3 + (-1.304e-07) * E**2 * t
        + 2.274e-08 * E * t**2 + 2.42e-07 * t**3
    )


def quantity_mulberry_paper(E, t):  # 桑皮纸 pH 8.3
    return _nonneg(
        1.637 - 0.03784 * E + 0.01934 * t + 0.0003198 * E**2
        + 5.837e-05 * E * t - 0.0001357 * t**2
        + (-7.016e-07) * E**3 + 2.139e-07 * E**2 * t
        + (-2.765e-07) * E * t**2 + 2.615e-07 * t**3
    )


def quantity_silk(E, t):  # 绢 pH 5.9
    return _nonneg(
        2.702 - 0.04803 * E + 0.04457 * t + 0.0002949 * E**2
        + 8.029e-05 * E * t - 0.0002707 * t**2
        + (-5.356e-07) * E**3 + (-2.034e-07) * E**2 * t
        + (-5.443e-08) * E * t**2 + 4.228e-07 * t**3
    )


# ---------------------------------------------------------------------------
# 光谱响应率 P(λ)（不裁剪负值）
# ---------------------------------------------------------------------------
def response_newsprint(wavelength):  # 新闻纸
    lam = np.asarray(wavelength, dtype=float)
    return 3.595e07 * lam ** (-2.902)


def response_bamboo_paper(wavelength):  # 竹纸
    lam = np.asarray(wavelength, dtype=float)
    return 1.0 + (-3.097e-24) * lam**7.979


def response_xuan_paper(wavelength):  # 宣纸
    lam = np.asarray(wavelength, dtype=float)
    return 37.42 - 0.1988 * lam + 0.000351 * lam**2 + (-2.03e-07) * lam**3


def response_hemp_paper(wavelength):  # 麻纸
    lam = np.asarray(wavelength, dtype=float)
    return 5.593 - 0.01859 * lam + (1.696e-05) * lam**2


def response_mulberry_paper(wavelength):  # 桑皮纸
    lam = np.asarray(wavelength, dtype=float)
    return -33.65 + 0.1287 * lam - 0.0001088 * lam**2


def response_silk(wavelength):  # 绢
    lam = np.asarray(wavelength, dtype=float)
    return 56.1 - 0.3057 * lam + 0.0005507 * lam**2 + (-3.233e-07) * lam**3


# =============================================================
# SUBSTRATE_MODEL_SECTION_END
# =============================================================
