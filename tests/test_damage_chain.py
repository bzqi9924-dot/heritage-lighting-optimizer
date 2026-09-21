# -*- coding: utf-8 -*-
"""验收项 T5–T7：公式单元测试、基准损伤、整体损伤手算一致。

基材公式已替换为《基材公式补充.docx》正式公式（决议 R10–R14）：
E 为照度（lx，等同项目其余部分的 I）、f 负值按 0 处理、pH 不再是计算变量。
"""
import numpy as np
import pytest

from models.damage_chain import (
    ModelCalculationError,
    normalized_damage,
    reference_damage,
    relic_damage,
)
from models.material_registry import MATERIALS, get_material
from utils.spectral_io import load_spectral_data


@pytest.fixture(scope="module")
def spectral():
    return load_spectral_data()


# T5 公式单元测试：f 与直接公式一致（颜料组）
def test_t5_f_formulas_pigments():
    cases = {
        "cochineal": (50.0, 100.0, 0.0214 * 100.0**0.3274 * 50.0**0.5710),
        "madder": (50.0, 100.0, 0.0589 * 100.0**0.2696 * 50.0**0.4532),
        "shellac": (80.0, 50.0, 0.0086 * 50.0**0.1258 * 80.0**0.9045),
        "sappanwood": (30.0, 200.0, 0.0061 * 200.0**0.1337 * 30.0**0.9337),
        "sophora": (60.0, 120.0, 0.0713 * 120.0**0.1190 * 60.0**0.6207),
        "gardenia": (45.0, 90.0, 0.0708 * 90.0**0.1598 * 45.0**0.5029),
        "phellodendron": (70.0, 150.0, 0.1957 * 150.0**0.1719 * 70.0**0.3163),
        "gamboge": (55.0, 80.0, 0.0230 * 80.0**0.1825 * 55.0**0.6840),
        "indigo": (65.0, 130.0, 0.0566 * 130.0**0.2295 * 65.0**0.4926),
        "realgar": (40.0, 110.0, 0.0248 * 40.0**0.8313 * 110.0**0.3957),
        "red_lead": (35.0, 95.0, 0.004723 * 35.0**0.4636 * 95.0**0.4655),
        "cinnabar": (50.0, 100.0, 0.0305 * 100.0**0.4274 * 50.0**0.9088),
    }
    for key, (E, t, expected) in cases.items():
        mat = get_material(key)
        assert np.isclose(mat.f_value(E, t), expected, rtol=1e-9, atol=1e-12), key


def test_t5_f_formulas_substrates():
    """六种基材照度—时间公式与《基材公式补充.docx》逐项一致（E=50, t=100）。"""
    E, t = 50.0, 100.0
    cases = {
        "newsprint": (
            -0.3421 + 0.01458 * E + 0.007503 * t - 0.0001127 * E**2
            + 6.322e-05 * E * t - 7.19e-05 * t**2
            + 2.452e-07 * E**3 + 1.762e-08 * E**2 * t
            - 1.627e-07 * E * t**2 + 1.523e-07 * t**3
        ),
        "bamboo_paper": (
            -0.3243 + 0.01883 * E + 0.04584 * t - 0.0001011 * E**2
            + 4.822e-05 * E * t - 0.0003088 * t**2
            + 1.779e-07 * E**3 + 9.369e-08 * E**2 * t
            - 1.632e-07 * E * t**2 + 5.324e-07 * t**3
        ),
        "xuan_paper": (
            -0.7135 + 0.01648 * E + 0.01581 * t + (-8.398e-05) * E**2
            + 2.313e-05 * E * t + (-5.914e-05) * t**2
            + 1.442e-07 * E**3 + 6.651e-08 * E**2 * t
            + (-1.188e-07) * E * t**2 + 8.27e-08 * t**3
        ),
        "hemp_paper": (
            3.147 - 0.06264 * E + 0.03753 * t + 0.0003611 * E**2
            + 5.717e-05 * E * t - 0.0001808 * t**2
            + (-5.994e-07) * E**3 + (-1.304e-07) * E**2 * t
            + 2.274e-08 * E * t**2 + 2.42e-07 * t**3
        ),
        "mulberry_paper": (
            1.637 - 0.03784 * E + 0.01934 * t + 0.0003198 * E**2
            + 5.837e-05 * E * t - 0.0001357 * t**2
            + (-7.016e-07) * E**3 + 2.139e-07 * E**2 * t
            + (-2.765e-07) * E * t**2 + 2.615e-07 * t**3
        ),
        "silk": (
            2.702 - 0.04803 * E + 0.04457 * t + 0.0002949 * E**2
            + 8.029e-05 * E * t - 0.0002707 * t**2
            + (-5.356e-07) * E**3 + (-2.034e-07) * E**2 * t
            + (-5.443e-08) * E * t**2 + 4.228e-07 * t**3
        ),
    }
    for key, expected in cases.items():
        mat = get_material(key)
        assert np.isclose(mat.f_value(E, t), expected, rtol=1e-9, atol=1e-12), key


def test_t5_substrate_f_nonnegative():
    """决议 R11：f(E,t) < 0 时按 0 处理（新闻纸/竹纸/宣纸在 E=1,t=1 原式为负）。"""
    raw = (-0.3421 + 0.01458 * 1 + 0.007503 * 1 - 0.0001127 * 1
           + 6.322e-05 * 1 * 1 - 7.19e-05 * 1
           + 2.452e-07 + 1.762e-08 - 1.627e-07 + 1.523e-07)
    assert raw < 0  # 原式为负
    assert get_material("newsprint").f_value(1.0, 1.0) == 0.0
    assert get_material("bamboo_paper").f_value(1.0, 1.0) == 0.0
    assert get_material("xuan_paper").f_value(1.0, 1.0) == 0.0


def test_t5_p_wavelength_response(spectral):
    """P(λ) 在 380-780 上可计算且为有限值（无机含定义域规则；负值不裁剪）。"""
    for mat in MATERIALS:
        P = mat.p_value(spectral.wavelength)
        assert P.shape == spectral.wavelength.shape
        assert np.all(np.isfinite(P)), mat.name_zh


def test_t5_substrate_p_formulas(spectral):
    """基材光谱响应率公式与文档一致（不裁剪负值）。"""
    lam = spectral.wavelength
    checks = {
        "newsprint": 3.595e07 * lam ** (-2.902),
        "bamboo_paper": 1.0 + (-3.097e-24) * lam**7.979,
        "xuan_paper": 37.42 - 0.1988 * lam + 0.000351 * lam**2 + (-2.03e-07) * lam**3,
        "hemp_paper": 5.593 - 0.01859 * lam + (1.696e-05) * lam**2,
        "mulberry_paper": -33.65 + 0.1287 * lam - 0.0001088 * lam**2,
        "silk": 56.1 - 0.3057 * lam + 0.0005507 * lam**2 + (-3.233e-07) * lam**3,
    }
    for key, expected in checks.items():
        got = get_material(key).p_value(lam)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, err_msg=key)
    # 文档给出的公式在部分波段确实为负，须保留负值（不裁剪）
    assert get_material("silk").p_value(lam)[-1] < 0


# T6 基准损伤：所有材料 D0 有限且不为 0（参考工况 E=50 lx, t=100 h）
def test_t6_reference_damage(spectral):
    for mat in MATERIALS:
        d0 = reference_damage(mat, spectral.wavelength, spectral.d55_normalized)
        assert np.isfinite(d0), mat.name_zh
        assert abs(d0) > 1e-12, mat.name_zh


# T7 整体损伤手算一致：两材料案例
def test_t7_relic_damage_manual(spectral):
    mats = [get_material("cochineal"), get_material("newsprint")]
    alpha = np.array([0.6, 0.4])
    I, t = 80.0, 120.0
    rng = np.random.default_rng(3)
    w = rng.dirichlet(np.ones(10))
    from utils.spectral_io import mix_led_channels

    S = mix_led_channels(spectral, w)

    total, rows = relic_damage(
        mats, alpha, spectral.wavelength, S, spectral.d55_normalized, I, t,
        details=True,
    )
    manual = 0.0
    for m, a in zip(mats, alpha):
        _, _, _, dn = normalized_damage(
            m, spectral.wavelength, S, spectral.d55_normalized, I, t
        )
        manual += a * dn
    assert np.isclose(total, manual, rtol=1e-9, atol=1e-12)
    assert set(rows[0].keys()) == {
        "material", "alpha", "K", "f", "D_raw", "D0", "D_norm", "alpha_D_norm"
    }
    assert np.isclose(sum(r["alpha_D_norm"] for r in rows), total, rtol=1e-9)


def test_t7_alpha_must_sum_to_one(spectral):
    mats = [get_material("cochineal")]
    with pytest.raises(ValueError):
        relic_damage(mats, np.array([0.5]), spectral.wavelength,
                     spectral.d55_normalized, spectral.d55_normalized, 50.0, 100.0)


def test_d0_zero_aborts():
    """D0 异常接近 0 必须中止（v3 第 6.7 节）。"""
    from types import SimpleNamespace

    fake = SimpleNamespace(
        name_zh="假材料",
        f_value=lambda E, t: 1.0,
        p_value=lambda wl: np.zeros_like(wl),  # P 全 0 -> D0 分母为 0
    )
    wl = np.arange(380, 781)
    with pytest.raises(ModelCalculationError):
        normalized_damage(fake, wl, np.ones_like(wl), np.ones_like(wl), 50.0, 100.0)
