# -*- coding: utf-8 -*-
"""9 种有机颜料的照度—时间模型 f 与波长响应函数 P(λ)。

数值来源：《公式集合_初版》/《产品说明 v3》第 7.1 / 7.2 节。
参数与函数形式不得修改；负值 P 不裁剪（决议 R8b）。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 照度—时间数量损伤 f(I, t)
# ---------------------------------------------------------------------------


def f_cochineal(I, t):  # 胭脂虫
    return 0.0214 * t**0.3274 * I**0.5710


def f_madder(I, t):  # 茜草
    return 0.0589 * t**0.2696 * I**0.4532


def f_shellac(I, t):  # 紫胶
    return 0.0086 * t**0.1258 * I**0.9045


def f_sappanwood(I, t):  # 苏木
    return 0.0061 * t**0.1337 * I**0.9337


def f_sophora(I, t):  # 槐米
    return 0.0713 * t**0.1190 * I**0.6207


def f_gardenia(I, t):  # 栀子
    return 0.0708 * t**0.1598 * I**0.5029


def f_phellodendron(I, t):  # 黄檗
    return 0.1957 * t**0.1719 * I**0.3163


def f_gamboge(I, t):  # 藤黄
    return 0.0230 * t**0.1825 * I**0.6840


def f_indigo(I, t):  # 花青
    return 0.0566 * t**0.2295 * I**0.4926


# ---------------------------------------------------------------------------
# 波长响应 P(λ)
# ---------------------------------------------------------------------------


def _cubic(c3, c2, c1, c0):
    def P(wavelength):
        lam = np.asarray(wavelength, dtype=float)
        return c3 * lam**3 + c2 * lam**2 + c1 * lam + c0

    return P


def _lorentz_sum(terms, offset):
    """terms: [(amplitude, center, width), ...]；offset: 常数项。"""

    def P(wavelength):
        lam = np.asarray(wavelength, dtype=float)
        out = np.full_like(lam, offset, dtype=float)
        for amp, center, width in terms:
            out = out + amp * (width**2 / ((lam - center) ** 2 + width**2))
        return out

    return P


P_cochineal = _cubic(2.6359e-7, -4.3136e-4, 0.2183, -29.95)
P_madder = _cubic(3.3471e-7, -5.6918e-4, 0.3067, -50.1022)
P_shellac = _cubic(2.7954e-7, -4.8989e-4, 0.2741, -47.3286)
P_sappanwood = _cubic(2.2787e-7, -4.0009e-4, 0.2266, -39.3298)
P_sophora = _lorentz_sum(
    [(2.9741, 454.17, 64.9012), (1.9282, 600.0, 100.0)], offset=1.1562
)
P_gardenia = _lorentz_sum(
    [(3.1884, 441.52, 32.8757), (1.3487, 560.43, 57.2318)], offset=1.2730
)
P_phellodendron = _lorentz_sum(
    [(2.6660, 425.18, 39.1921), (1.2657, 565.50, 81.7925)], offset=0.9259
)
P_gamboge = _lorentz_sum(
    [(4.1568, 432.26, 25.8397), (1.6663, 574.49, 58.3282)], offset=1.8630
)
P_indigo = _cubic(1.8967e-7, -3.2383e-4, 0.1735, -26.1221)
