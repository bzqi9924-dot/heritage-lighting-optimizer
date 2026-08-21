# -*- coding: utf-8 -*-
"""验收项 T5–T7：公式单元测试、基准损伤、整体损伤手算一致。"""
import numpy as np
import pytest

from models.damage_chain import (
    ModelCalculationError,
    material_detail,
    normalized_damage,
    reference_damage,
    relic_damage,
)
from models.material_registry import MATERIALS, get_material
from utils.spectral_io import load_spectral_data


@pytest.fixture(scope="module")
def spectral():
    return load_spectral_data()


# T5 公式单元测试：f 与直接公式一致（每组材料选 I, t[,pH]）
def test_t5_f_formulas():
    cases = {
        # key: (I, t, expected via direct formula)
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
        # 基材（含 pH，宣纸按决议 R3 用 pH^0.2）
        "hemp_paper": (50.0, 100.0, 0.023 * 50.0**0.213 * 100.0**0.432 * 7.0**0.2),
        "mulberry_paper": (50.0, 100.0, 0.044 * 50.0**0.857 * 100.0**0.543 * 7.0**0.2),
        "newsprint": (50.0, 100.0, 0.056 * 50.0**0.234 * 100.0**0.234 * 7.0**0.2),
        "bamboo_paper": (50.0, 100.0, 0.038 * 50.0**0.456 * 100.0**0.1223 * 7.0**0.2),
    }
    for key, (I, t, expected) in cases.items():
        mat = get_material(key)
        got = mat.f_value(I, t, pH=7.0 if mat.requires_ph else None)
        assert np.isclose(got, expected, rtol=1e-9, atol=1e-12), key


def test_t5_xuan_paper_formula():
    """宣纸多项式 + pH^0.2（决议 R3）。"""
    mat = get_material("xuan_paper")
    I, t, pH = 50.0, 100.0, 7.0
    expected = (
        -7.51e-11 - 1.83e-6 * t + 4.19e-10 * t**2
        - 9.41e-5 * I + 1.6e-6 * I**2 - 6.58e-9 * I**3
        + 3.61e-7 * t * I - 4.6e-9 * t * I**2
        - 5.76e-10 * t**2 * I + 5.18e-12 * t**2 * I**2
        + pH**0.2
    )
    assert np.isclose(mat.f_value(I, t, pH=pH), expected, rtol=1e-9, atol=1e-12)


def test_t5_p_wavelength_response(spectral):
    """P(λ) 在 380-780 上可计算且为有限值（无机含定义域规则）。"""
    for mat in MATERIALS:
        P = mat.p_value(spectral.wavelength)
        assert P.shape == spectral.wavelength.shape
        assert np.all(np.isfinite(P)), mat.name_zh


# T6 基准损伤：所有材料 D0 有限且不为 0
def test_t6_reference_damage(spectral):
    for mat in MATERIALS:
        d0 = reference_damage(mat, spectral.wavelength, spectral.d55_normalized,
                              pH=7.0 if mat.requires_ph else None)
        assert np.isfinite(d0), mat.name_zh
        assert abs(d0) > 1e-12, mat.name_zh


# T7 整体损伤手算一致：两材料案例
def test_t7_relic_damage_manual(spectral):
    mats = [get_material("cochineal"), get_material("madder")]
    alpha = np.array([0.6, 0.4])
    I, t = 80.0, 120.0
    rng = np.random.default_rng(3)
    w = rng.dirichlet(np.ones(10))
    S = None  # 直接构造候选（与 mix 一致即可）
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
    # 明细行字段完整
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
        requires_ph=False,
        f_value=lambda I, t, pH=None: 1.0,
        p_value=lambda wl: np.zeros_like(wl),  # P 全 0 -> D0 分母为 0
    )
    wl = np.arange(380, 781)
    with pytest.raises(ModelCalculationError):
        normalized_damage(fake, wl, np.ones_like(wl), np.ones_like(wl), 50.0, 100.0)
