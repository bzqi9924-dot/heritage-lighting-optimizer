# -*- coding: utf-8 -*-
"""损伤计算链：K(S)、D_raw、D0（统一参考工况）、D_norm、D_relic。

口径：《产品说明 v3》第 6.3–6.8 节。
- K_m(S_ω) = ∫S_ω·P_m / ∫S_0·P_m（梯形积分，显式传波长轴）；
- D_raw = K_m · f_m(I,t[,pH])；
- D0 通过同一个 raw_damage() 通用函数计算（S0=D55, I0=50 lx, t0=100 h），
  基材分子与基准分母使用同一个用户输入 pH（封装在本文件单独函数中，
  标注"随正式基材模型可替换"）；
- D_norm = D_raw / D0；D0 为 0/非有限/接近 0 时中止该材料计算并报错；
- D_relic = Σ α_m · D_norm,m。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from config import REFERENCE_ILLUMINANCE, REFERENCE_TIME
from models.material_registry import MaterialSpec


class ModelCalculationError(Exception):
    """模型数值异常（K 分母、D0 分母等），对应 v3 第 6.3 / 6.7 / 14 节。"""


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """梯形积分，兼容 numpy 1.x (trapz) 与 2.x (trapezoid)。"""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def spectral_correction(
    wavelength: np.ndarray,
    S_candidate: np.ndarray,
    S_d55: np.ndarray,
    P_material: np.ndarray,
) -> float:
    """材料光谱修正因子 K_m(S_ω)（v3 第 6.3 节）。"""
    S_candidate = np.asarray(S_candidate, dtype=float)
    S_d55 = np.asarray(S_d55, dtype=float)
    P_material = np.asarray(P_material, dtype=float)
    numerator = _trapz(S_candidate * P_material, wavelength)
    denominator = _trapz(S_d55 * P_material, wavelength)
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        raise ModelCalculationError("Invalid D55 response denominator")
    return numerator / denominator


def raw_damage(
    material: MaterialSpec,
    wavelength: np.ndarray,
    S_candidate: np.ndarray,
    S_d55: np.ndarray,
    I: float,
    t: float,
    pH: float | None = None,
) -> float:
    """单材料原始损伤 D_raw,m = K_m(S_ω) · f_m(I,t[,pH])（v3 第 6.5 节）。"""
    P = material.p_value(wavelength)
    K = spectral_correction(wavelength, S_candidate, S_d55, P)
    f = material.f_value(I, t, pH)
    return K * f


def reference_damage(
    material: MaterialSpec,
    wavelength: np.ndarray,
    S_d55: np.ndarray,
    pH: float | None = None,
    I0: float = REFERENCE_ILLUMINANCE,
    t0: float = REFERENCE_TIME,
) -> float:
    """统一参考工况损伤基准 D_0,m（v3 第 6.6 节）。

    通过同一个通用 raw_damage() 计算，不手工写死表值。基材分子与基准分母
    使用同一个用户输入 pH。随正式基材模型可替换：本函数无需改动，
    依赖 material_registry 分派的 f/P。
    """
    return raw_damage(material, wavelength, S_d55, S_d55, I0, t0, pH)


def normalized_damage(
    material: MaterialSpec,
    wavelength: np.ndarray,
    S_candidate: np.ndarray,
    S_d55: np.ndarray,
    I: float,
    t: float,
    pH: float | None = None,
) -> tuple[float, float, float, float]:
    """D_norm,m = D_raw,m / D_0,m，并返回中间量 (K, D_raw, D0, D_norm)。

    D0 为 0 / 非有限 / 异常接近 0 时中止计算并报错，不得把分母替换为 1
    （v3 第 6.7 节）。
    """
    P = material.p_value(wavelength)
    K = spectral_correction(wavelength, S_candidate, S_d55, P)
    f = material.f_value(I, t, pH)
    d_raw = K * f
    d0 = reference_damage(material, wavelength, S_d55, pH)
    if not np.isfinite(d0) or abs(d0) < 1e-12:
        raise ModelCalculationError(
            f"材料 {material.name_zh} 参考损伤 D0 为 0/非有限/接近 0，已中止计算"
        )
    d_norm = d_raw / d0
    if not np.isfinite(d_norm):
        raise ModelCalculationError(
            f"材料 {material.name_zh} 无量纲损伤非有限，已中止计算"
        )
    return K, d_raw, d0, d_norm


def material_detail(
    material: MaterialSpec,
    wavelength: np.ndarray,
    S_candidate: np.ndarray,
    S_d55: np.ndarray,
    I: float,
    t: float,
    pH: float | None,
    alpha: float,
) -> dict:
    """单材料明细（用于界面"计算明细"折叠区，v3 第 12 节）。"""
    K, d_raw, d0, d_norm = normalized_damage(
        material, wavelength, S_candidate, S_d55, I, t, pH
    )
    return {
        "material": material.name_zh,
        "alpha": alpha,
        "K": K,
        "f": d_raw / K if K != 0 else float("nan"),
        "D_raw": d_raw,
        "D0": d0,
        "D_norm": d_norm,
        "alpha_D_norm": alpha * d_norm,
    }


def relic_damage(
    materials: list[MaterialSpec],
    alpha: np.ndarray,
    wavelength: np.ndarray,
    S_candidate: np.ndarray,
    S_d55: np.ndarray,
    I: float,
    t: float,
    pH: float | None = None,
    details: bool = False,
) -> float | tuple[float, list[dict]]:
    """整件文物整体损伤 D_relic = Σ α_m · D_norm,m（v3 第 6.8 节）。"""
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim != 1 or alpha.shape[0] != len(materials):
        raise ValueError("alpha 个数必须等于选中材料数")
    if np.any(alpha < 0) or abs(float(np.sum(alpha)) - 1.0) > 1e-6:
        raise ValueError(f"面积比例必须非负且总和为 1，当前总和 {float(np.sum(alpha)):.6f}")

    total = 0.0
    detail_rows: list[dict] = []
    for m, a in zip(materials, alpha):
        if details:
            row = material_detail(m, wavelength, S_candidate, S_d55, I, t, pH, float(a))
            detail_rows.append(row)
            total += row["alpha_D_norm"]
        else:
            _, _, _, d_norm = normalized_damage(
                m, wavelength, S_candidate, S_d55, I, t, pH
            )
            total += float(a) * d_norm
    if details:
        return total, detail_rows
    return total
