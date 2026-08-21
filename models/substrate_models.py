# -*- coding: utf-8 -*-
"""五种纸质基材的照度—时间模型 f(I,t,pH) 与波长响应函数 P(λ)。

# =============================================================
# REPLACEABLE_SUBSTRATE_MODEL_START
# 当前版本：来自《公式集合_初版》；仅用于现阶段软件联调与论文应用框架。
# 后续正式模型到位时，仅替换本区内对应 quantity_* 与 response_* 函数。
# 不修改 UI、damage_chain.py、optimizer.py。
# =============================================================

数值口径：
- 除宣纸外四种基材 f = k·I^a·t^b·pH^0.2；
- 宣纸 f 为多项式，其末尾 pH 指数按《需求确认记录》决议 R3 由 pH^2 视为笔误
  改为 pH^0.2（与其他基材一致）。代码处标注 TODO_XUAN_PAPER_PH_EXPONENT，
  供后续核正式公式时确认。
- P(λ) 负值不裁剪、不取绝对值（决议 R8b；竹纸 P 在 380–780 nm 全区间为负）。
"""
from __future__ import annotations

import numpy as np


def quantity_hemp_paper(I, t, pH):  # 麻纸  TODO_REPLACE_SUBSTRATE_MODEL_HEMP
    return 0.023 * I**0.213 * t**0.432 * pH**0.2


def quantity_mulberry_paper(I, t, pH):  # 桑纸  TODO_REPLACE_SUBSTRATE_MODEL_MULBERRY
    return 0.044 * I**0.857 * t**0.543 * pH**0.2


def quantity_newsprint(I, t, pH):  # 新闻纸  TODO_REPLACE_SUBSTRATE_MODEL_NEWSPRINT
    return 0.056 * I**0.234 * t**0.234 * pH**0.2


def quantity_xuan_paper(I, t, pH):  # 宣纸  TODO_REPLACE_SUBSTRATE_MODEL_XUAN
    # 多项式部分（不含 pH 项），按《公式集合_初版》逐项照抄：
    #   -7.51e-11 -1.83e-6·t +4.19e-10·t^2 -9.41e-5·I +1.6e-6·I^2 -6.58e-9·I^3
    #   +3.61e-7·t·I -4.6e-9·t·I^2 -5.76e-10·t^2·I +5.18e-12·t^2·I^2
    poly = (
        -7.51e-11
        - 1.83e-6 * t
        + 4.19e-10 * t**2
        - 9.41e-5 * I
        + 1.6e-6 * I**2
        - 6.58e-9 * I**3
        + 3.61e-7 * t * I
        - 4.6e-9 * t * I**2
        - 5.76e-10 * t**2 * I
        + 5.18e-12 * t**2 * I**2
    )
    # TODO_XUAN_PAPER_PH_EXPONENT：决议 R3 —— 原公式为 "+pH^2"，视为笔误，
    # 改为 "+pH^0.2" 以与其他基材一致；正式公式核正后仅改此处指数。
    return poly + pH**0.2


def quantity_bamboo_paper(I, t, pH):  # 竹纸  TODO_REPLACE_SUBSTRATE_MODEL_BAMBOO
    return 0.038 * I**0.456 * t**0.1223 * pH**0.2


def _substrate_cubic(c3, c2, c1, c0):
    def P(wavelength):
        lam = np.asarray(wavelength, dtype=float)
        return c3 * lam**3 + c2 * lam**2 + c1 * lam + c0

    return P


def response_hemp_paper(wavelength):  # 麻纸
    return _substrate_cubic(2.63e-7, -4.3136e-4, 0.2183, -29.0)(wavelength)


def response_mulberry_paper(wavelength):  # 桑纸
    return _substrate_cubic(3.24e-7, -4.3136e-4, 0.2183, -32.0)(wavelength)


def response_newsprint(wavelength):  # 新闻纸
    return _substrate_cubic(4.21e-7, -4.3136e-4, 0.2183, -35.0)(wavelength)


def response_xuan_paper(wavelength):  # 宣纸（四次多项式）
    lam = np.asarray(wavelength, dtype=float)
    return -405.1 + 2.87 * lam - 7.5e-3 * lam**2 + 8.64e-6 * lam**3 - 3.67e-9 * lam**4


def response_bamboo_paper(wavelength):  # 竹纸
    return _substrate_cubic(1.56e-7, -4.3136e-4, 0.2183, -52.0)(wavelength)


# =============================================================
# REPLACEABLE_SUBSTRATE_MODEL_END
# =============================================================
