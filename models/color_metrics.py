# -*- coding: utf-8 -*-
"""颜色质量指标：TM-30 Rf/Rg、CCT、Duv（统一封装）。

口径：《产品说明 v3》第 9 节 + 《需求确认记录》决议 R6。
- 主后端：colour-science（colour.quality.tm3018，TM-30-18 官方实现，含
  R_f/R_g/CCT/D_uv 输出）；
- 兜底后端：luxpy（spd_to_ies_tm30_metrics，TM-30-18 独立实现）；
- 两个后端结果一致（已在固定测试 SPD 上交叉校验，测试见 tests/）。
- 不得用 CRI Ra 代替 Rf/Rg，不得伪造指标；两库均不可用时明确报错并
  说明所需依赖（v3 第 16 节）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ColorMetricsError(Exception):
    """颜色指标计算失败（依赖缺失或数值异常）。"""


@dataclass(frozen=True)
class ColorMetricsResult:
    Rf: float
    Rg: float
    CCT: float   # K
    Duv: float


def _via_colour(wavelength: np.ndarray, spd: np.ndarray) -> ColorMetricsResult:
    from colour.quality.tm3018 import colour_fidelity_index_ANSIIESTM3018
    import colour

    sd = colour.SpectralDistribution(dict(zip(np.asarray(wavelength), np.asarray(spd))))
    spec = colour_fidelity_index_ANSIIESTM3018(sd, additional_data=True)
    return ColorMetricsResult(
        Rf=float(spec.R_f),
        Rg=float(spec.R_g),
        CCT=float(spec.CCT),
        Duv=float(spec.D_uv),
    )


def _via_luxpy(wavelength: np.ndarray, spd: np.ndarray) -> ColorMetricsResult:
    from luxpy.color.cri import spd_to_ies_tm30_metrics

    res = spd_to_ies_tm30_metrics(np.vstack([np.asarray(wavelength), np.asarray(spd)]))
    return ColorMetricsResult(
        Rf=float(res["Rf"][0, 0]),
        Rg=float(res["Rg"][0, 0]),
        CCT=float(res["cct"][0, 0]),
        Duv=float(res["duv"][0, 0]),
    )


_BACKEND: str | None = None  # "colour" / "luxpy"


def backend_name() -> str:
    """返回当前生效的颜色指标后端。"""
    global _BACKEND
    if _BACKEND is None:
        _detect_backend()
    return _BACKEND or "unavailable"


def _detect_backend() -> None:
    global _BACKEND
    try:
        import colour  # noqa: F401
        from colour.quality.tm3018 import colour_fidelity_index_ANSIIESTM3018  # noqa: F401

        _BACKEND = "colour"
        return
    except ImportError:
        pass
    try:
        from luxpy.color.cri import spd_to_ies_tm30_metrics  # noqa: F401

        _BACKEND = "luxpy"
        return
    except ImportError:
        pass
    _BACKEND = None


def compute_color_metrics(wavelength: np.ndarray, spd: np.ndarray) -> ColorMetricsResult:
    """计算候选光源 S_ω(λ) 的 Rf/Rg/CCT/Duv。"""
    global _BACKEND
    if _BACKEND is None:
        _detect_backend()
    spd = np.asarray(spd, dtype=float)
    if spd.ndim != 1 or spd.shape[0] != np.asarray(wavelength).shape[0]:
        raise ColorMetricsError("SPD 与波长轴长度不一致")
    if not np.all(np.isfinite(spd)):
        raise ColorMetricsError("SPD 包含 NaN/Inf，无法计算颜色指标")

    if _BACKEND == "colour":
        result = _via_colour(wavelength, spd)
    elif _BACKEND == "luxpy":
        result = _via_luxpy(wavelength, spd)
    else:
        raise ColorMetricsError(
            "缺少 TM-30 依赖：请安装 colour-science 或 luxpy "
            "（pip install colour-science luxpy），否则无法计算 Rf/Rg。"
        )
    if not all(np.isfinite([result.Rf, result.Rg, result.CCT, result.Duv])):
        raise ColorMetricsError("颜色指标计算结果为非有限值")
    return result
