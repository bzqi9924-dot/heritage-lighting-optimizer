# -*- coding: utf-8 -*-
"""验收项 T1–T4：光谱加载、最大值归一化、权重修复、K 自检。"""
import numpy as np
import pytest

from config import NUM_LED_CHANNELS, WAVELENGTH_MAX, WAVELENGTH_MIN
from models.material_registry import MATERIALS
from models.damage_chain import spectral_correction
from models.optimizer import WeightRepair, WeightSampling
from utils.spectral_io import DataValidationError, load_spectral_data, mix_led_channels


@pytest.fixture(scope="module")
def spectral():
    return load_spectral_data()


# T1 光谱加载
def test_t1_load_coverage_and_channels(spectral):
    assert spectral.n_channels == NUM_LED_CHANNELS == 10
    assert spectral.wavelength[0] <= WAVELENGTH_MIN + 1e-9
    assert spectral.wavelength[-1] >= WAVELENGTH_MAX - 1e-9
    assert spectral.d55_raw.shape == (spectral.wavelength.size,)
    assert spectral.led_raw.shape == (10, spectral.wavelength.size)
    assert np.all(np.isfinite(spectral.led_raw))
    assert np.all(np.isfinite(spectral.d55_raw))


# T2 最大值归一化
def test_t2_max_normalization(spectral):
    assert np.allclose(np.max(spectral.led_normalized, axis=1), 1.0, atol=1e-9)
    assert np.isclose(np.max(spectral.d55_normalized), 1.0, atol=1e-9)
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = rng.dirichlet(np.ones(10))
        S = mix_led_channels(spectral, w)
        assert np.isclose(np.max(S), 1.0, atol=1e-9)
        assert np.all(np.isfinite(S))


# T3 权重修复
def test_t3_weight_repair():
    problem = type("P", (), {"n_var": 11})()
    repair = WeightRepair()
    # 随机个体（可能含负权重）
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 11))
    X[:, 10] = rng.uniform(-5, 400, size=20)
    Xr = repair._do(problem, X.copy())
    W = Xr[:, :10]
    assert np.all(W >= 0)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(Xr[:, 10] >= 1.0) and np.all(Xr[:, 10] <= 300.0)
    # 全零权重 -> 重置为等权
    X0 = np.zeros((2, 11))
    X0[:, 10] = 50.0
    X0r = repair._do(problem, X0.copy())
    assert np.allclose(X0r[:, :10], 0.1)


def test_t3_weight_sampling():
    problem = type("P", (), {"n_var": 11})()
    s = WeightSampling(seed=42)
    X = s._do(problem, 50)
    assert X.shape == (50, 11)
    assert np.all(X[:, :10] >= 0)
    assert np.allclose(X[:, :10].sum(axis=1), 1.0, atol=1e-9)
    assert np.all(X[:, 10] >= 1.0) and np.all(X[:, 10] <= 300.0)


# T4 K 自检：候选光源 = D55 时 K ≈ 1
def test_t4_k_selfcheck(spectral):
    for mat in MATERIALS:
        P = mat.p_value(spectral.wavelength)
        K = spectral_correction(
            spectral.wavelength,
            spectral.d55_normalized,
            spectral.d55_normalized,
            P,
        )
        assert np.isfinite(K), mat.name_zh
        assert abs(K - 1.0) < 1e-9, f"{mat.name_zh}: K={K}"


# 光谱校验错误处理（v3 第 14 节）
def test_spectral_errors(tmp_path):
    import openpyxl

    # 波长范围不足 380-780
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([None] + [f"C{i}" for i in range(10)])
    for wl in range(400, 700):
        ws.append([wl] + [0.1] * 10)
    p = tmp_path / "bad_led.xlsx"
    wb.save(p)
    with pytest.raises(DataValidationError):
        load_spectral_data(led_file=p)

    # 通道数不是 10
    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    for wl in range(380, 781):
        ws2.append([wl, 0.1, 0.2])
    p2 = tmp_path / "few_ch.xlsx"
    wb2.save(p2)
    with pytest.raises(DataValidationError):
        load_spectral_data(led_file=p2)
