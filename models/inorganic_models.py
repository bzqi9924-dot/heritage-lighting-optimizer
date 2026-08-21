# -*- coding: utf-8 -*-
"""3 种无机颜料的照度—时间模型 f 与波长响应函数 P(λ)。

数值来源：《公式集合_初版》/《产品说明 v3》第 7.1 / 7.2 节。

对数定义域处理（v3 第 7.2 节 + 开发约定）：
    三个 P(λ) 均含 ln() 项，ln 自变量必须 > 0。当前初版模型未说明定义域之外
    的处理方式。为避免 NaN/崩溃，按产品说明临时约定集中封装：
    仅当 log_argument > 0 时按公式计算，定义域之外返回 0。
    TODO_CONFIRM_INORGANIC_LOG_DOMAIN：后续若模型来源给出不同定义，只修改
    本文件内的 _log_domain_lorentz，不改变优化器。
"""
from __future__ import annotations

import numpy as np


def _log_domain_lorentz(skew, center, width, ln2_numerator=1.0):
    """广义对数洛伦兹形式：

    P(λ) = exp{ -[ln2/(s^2)] · [ln(1 + 2·s·(λ-c)/w)]^2 }
    """
    def P(wavelength):
        lam = np.asarray(wavelength, dtype=float)
        arg = 1.0 + 2.0 * skew * (lam - center) / width
        out = np.zeros_like(lam, dtype=float)
        mask = arg > 0.0
        ln_arg = np.log(arg[mask])
        out[mask] = np.exp(
            -(np.log(2.0) * ln2_numerator / (skew**2)) * ln_arg**2
        )
        return out

    return P


# ---------------------------------------------------------------------------
# 照度—时间数量损伤 f(I, t)
# ---------------------------------------------------------------------------


def f_realgar(I, t):  # 雄黄
    return 0.0248 * I**0.8313 * t**0.3957


def f_red_lead(I, t):  # 铅丹
    return 0.004723 * I**0.4636 * t**0.4655


def f_cinnabar(I, t):  # 朱砂
    return 0.0305 * t**0.4274 * I**0.9088


# ---------------------------------------------------------------------------
# 波长响应 P(λ)（含定义域规则）
# ---------------------------------------------------------------------------
P_realgar = _log_domain_lorentz(skew=-0.9500, center=564.2, width=138.3)
P_red_lead = _log_domain_lorentz(skew=-0.9517, center=437.376, width=123.5)
P_cinnabar = _log_domain_lorentz(skew=0.1189, center=410.191, width=286.3)
