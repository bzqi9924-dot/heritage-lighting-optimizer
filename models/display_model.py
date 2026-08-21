# -*- coding: utf-8 -*-
"""展示效果模型 F（冷/暖/中三色调）。

口径：《产品说明 v3》第 7.3 节。公式系数照抄（决议 R8c：Rf 系数为负属
公式既定行为，不做修正）。
"""
from __future__ import annotations

TONE_COLD = "cold"
TONE_WARM = "warm"
TONE_NEUTRAL = "neutral"

TONE_ZH = {
    TONE_COLD: "冷色调",
    TONE_WARM: "暖色调",
    TONE_NEUTRAL: "中性色调",
}


def F_cold(I, Rf, Rg):
    """冷色调展示效果。"""
    return 0.026 * I - 0.004 * Rf - 0.006 * Rg + 6.300


def F_warm(I, Rf, Rg):
    """暖色调展示效果。"""
    return 0.023 * I - 0.013 * Rf + 0.004 * Rg + 6.172


def F_neutral(I, Rf, Rg):
    """中性色调展示效果。"""
    return 0.017 * I - 0.017 * Rf + 0.003 * Rg + 6.656


_FUNCS = {
    TONE_COLD: F_cold,
    TONE_WARM: F_warm,
    TONE_NEUTRAL: F_neutral,
}


def display_score(tone: str, I, Rf, Rg) -> float:
    """按色调类型返回展示效果 F。tone 支持英文键或中文名。"""
    if tone in _FUNCS:
        key = tone
    elif tone in TONE_ZH.values():
        key = next(k for k, v in TONE_ZH.items() if v == tone)
    else:
        raise ValueError(f"未知色调类型: {tone}，可选 {list(TONE_ZH.values())}")
    return float(_FUNCS[key](I, Rf, Rg))
