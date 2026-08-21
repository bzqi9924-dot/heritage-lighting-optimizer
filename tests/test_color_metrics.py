# -*- coding: utf-8 -*-
"""验收项 T8：展示模型 F 与直接公式一致；颜色指标验证（固定测试 SPD 有限、
D65 基准、colour/luxpy 双后端交叉一致）。"""
import numpy as np
import pytest

from models.color_metrics import (
    ColorMetricsError,
    backend_name,
    compute_color_metrics,
)
from models.display_model import F_cold, F_neutral, F_warm, display_score


# T8 展示模型：三种 F 在固定 I, Rf, Rg 下与直接公式一致
def test_t8_display_model():
    I, Rf, Rg = 100.0, 85.0, 100.0
    assert np.isclose(display_score("cold", I, Rf, Rg),
                      0.026 * I - 0.004 * Rf - 0.006 * Rg + 6.300)
    assert np.isclose(display_score("warm", I, Rf, Rg),
                      0.023 * I - 0.013 * Rf + 0.004 * Rg + 6.172)
    assert np.isclose(display_score("neutral", I, Rf, Rg),
                      0.017 * I - 0.017 * Rf + 0.003 * Rg + 6.656)
    assert np.isclose(F_cold(I, Rf, Rg), display_score("冷色调", I, Rf, Rg))
    assert np.isclose(F_warm(I, Rf, Rg), display_score("暖色调", I, Rf, Rg))
    assert np.isclose(F_neutral(I, Rf, Rg), display_score("中性色调", I, Rf, Rg))
    with pytest.raises(ValueError):
        display_score("unknown", I, Rf, Rg)


@pytest.fixture(scope="module")
def test_spds():
    """固定测试 SPD 集：D65、A、暖白 LED 混合、冷白 LED 混合。"""
    import colour

    wl = np.arange(380, 781)
    out = {}
    for name in ["D65", "A"]:
        sd = colour.SDS_ILLUMINANTS[name].copy().align(colour.SpectralShape(380, 780, 1))
        out[name] = np.interp(wl, sd.wavelengths, sd.values)
        out[name] /= out[name].max()
    rng = np.random.default_rng(7)
    for label, centers in [("warm_mix", [593, 605, 629, 665]),
                           ("cool_mix", [448, 469, 504, 523])]:
        s = np.zeros_like(wl, dtype=float)
        for c in centers:
            s += rng.uniform(0.3, 1.0) * np.exp(-((wl - c) / 20.0) ** 2)
        out[label] = s / s.max()
    return wl, out


def test_backend_available():
    assert backend_name() in ("colour", "luxpy")


def test_metrics_finite_on_fixed_spds(test_spds):
    wl, spds = test_spds
    for name, spd in spds.items():
        m = compute_color_metrics(wl, spd)
        for v in (m.Rf, m.Rg, m.CCT, m.Duv):
            assert np.isfinite(v), name


def test_d65_reference_values(test_spds):
    """D65：Rf≈100、Rg≈100、CCT≈6504 K（Ohno 口径下 Duv≈+0.0032）。"""
    m = compute_color_metrics(test_spds[0], test_spds[1]["D65"])
    assert abs(m.Rf - 100.0) < 0.2
    assert abs(m.Rg - 100.0) < 0.5
    assert abs(m.CCT - 6504.0) < 30.0
    assert abs(m.Duv - 0.00321) < 0.001


def test_backends_agree(test_spds):
    """colour 与 luxpy 两个后端在近白光源上结果一致（若两者均可用）。

    注：对远离白点的窄带光源（如纯蓝 LED 混合），CCT/Duv 估计病态且与实现
    细节相关（不同算法/色温轨迹），两库可能给出不同 CCT——这类光源本就会被
    强约束拒绝，不影响应用。故交叉校验仅针对白光区间的测试 SPD。
    """
    try:
        import colour.quality.tm3018  # noqa: F401
        import luxpy  # noqa: F401
    except ImportError:
        pytest.skip("需要 colour-science 与 luxpy 双后端")
    from models import color_metrics as cm

    wl, spds = test_spds
    for name in ["D65", "A", "warm_mix"]:
        m_colour = cm._via_colour(wl, spds[name])
        m_luxpy = cm._via_luxpy(wl, spds[name])
        assert abs(m_colour.Rf - m_luxpy.Rf) < 0.5, name
        assert abs(m_colour.Rg - m_luxpy.Rg) < 0.8, name
        assert abs(m_colour.CCT - m_luxpy.CCT) < 50.0, name
        assert abs(m_colour.Duv - m_luxpy.Duv) < 0.0005, name


def test_bad_spd_raises():
    with pytest.raises(ColorMetricsError):
        compute_color_metrics(np.arange(380, 781), np.full(401, np.nan))
