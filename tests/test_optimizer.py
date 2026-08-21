# -*- coding: utf-8 -*-
"""验收项 T9–T11、T13：强约束、代表方案、可复现性、界面隔离。"""
from pathlib import Path

import numpy as np
import pytest

import models.optimizer as opt
from config import ILLUMINANCE_MAX, ILLUMINANCE_MIN
from models.color_metrics import ColorMetricsError
from models.material_registry import get_material
from models.optimizer import NoFeasibleSolutionError, run_optimization
from utils.spectral_io import load_spectral_data

ROOT = Path(__file__).resolve().parent.parent

POP, GEN, SEED = 30, 20, 42


@pytest.fixture(scope="module")
def spectral():
    return load_spectral_data()


@pytest.fixture(scope="module")
def setup():
    spectral = load_spectral_data()
    mats = [get_material("cochineal"), get_material("madder"), get_material("hemp_paper")]
    alpha = np.array([0.4, 0.3, 0.3])
    return spectral, mats, alpha


@pytest.fixture(scope="module")
def result(setup):
    spectral, mats, alpha = setup
    return run_optimization(spectral, mats, alpha, t=500.0, tone="warm", pH=7.0,
                            pop_size=POP, n_gen=GEN, seed=SEED)


# T9 强约束：最终 Pareto 每个方案均满足 Rf/Rg/CCT/Duv、I 与权重约束
def test_t9_all_pareto_feasible(result):
    df = result.pareto_df
    assert len(df) > 0
    assert df["Rf"].ge(70 - 1e-6).all()
    assert df["Rg"].ge(88 - 1e-6).all()
    assert df["CCT_K"].between(2650 - 1e-6, 5550 + 1e-6).all()
    assert df["Duv"].abs().le(0.0054 + 1e-6).all()
    assert df["I_lx"].between(ILLUMINANCE_MIN - 1e-6, ILLUMINANCE_MAX + 1e-6).all()
    w = df[[f"ω{q+1}" for q in range(10)]].to_numpy()
    assert np.all(w >= 0)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)


# T10 代表方案
def test_t10_representatives(result):
    df = result.pareto_df
    reps = result.representatives

    i_prot = int(df["D_relic"].idxmin())
    assert reps["保护优先"].D_relic == pytest.approx(float(df["D_relic"].min()), abs=1e-9)

    i_disp = int(df["F"].idxmax())
    assert reps["展示优先"].F == pytest.approx(float(df["F"].max()), abs=1e-9)

    # 综合折中（决议 R9）：最小最大归一化偏差
    #   dev_D = (D-Dmin)/(Dmax-Dmin)；dev_F = (Fmax-F)/(Fmax-Fmin)
    #   取 max(dev_D, dev_F) 最小的方案
    d = df["D_relic"].to_numpy()
    f = df["F"].to_numpy()
    d_norm = (d - d.min()) / (d.max() - d.min())
    f_norm = (f - f.min()) / (f.max() - f.min())
    deviation = np.maximum(d_norm, 1.0 - f_norm)
    i_comp = int(np.argmin(deviation))
    assert reps["综合折中"].D_relic == pytest.approx(float(d[i_comp]), abs=1e-9)
    assert reps["综合折中"].F == pytest.approx(float(f[i_comp]), abs=1e-9)

    # 代表方案明细字段完整
    for rep in reps.values():
        assert set(rep.details[0].keys()) == {
            "material", "alpha", "K", "f", "D_raw", "D0", "D_norm", "alpha_D_norm"
        }


# T11 可复现性：相同 seed 与参数下结果可重复
def test_t11_reproducibility(setup):
    spectral, mats, alpha = setup
    r1 = run_optimization(spectral, mats, alpha, t=500.0, tone="warm", pH=7.0,
                          pop_size=POP, n_gen=GEN, seed=SEED)
    r2 = run_optimization(spectral, mats, alpha, t=500.0, tone="warm", pH=7.0,
                          pop_size=POP, n_gen=GEN, seed=SEED)
    assert len(r1.pareto_df) == len(r2.pareto_df)
    np.testing.assert_allclose(
        r1.pareto_df[["D_relic", "F"]].to_numpy(),
        r2.pareto_df[["D_relic", "F"]].to_numpy(),
        atol=1e-9,
    )
    for k in r1.representatives:
        assert r1.representatives[k].D_relic == pytest.approx(
            r2.representatives[k].D_relic, abs=1e-9
        )


# 无可行解时给出提示而非伪造结果
def test_no_feasible_solution(setup, monkeypatch):
    spectral, mats, alpha = setup

    def _boom(wavelength, spd):
        raise ColorMetricsError("synthetic failure")

    monkeypatch.setattr(opt, "compute_color_metrics", _boom)
    with pytest.raises(NoFeasibleSolutionError):
        run_optimization(spectral, mats, alpha, t=500.0, tone="warm", pH=7.0,
                         pop_size=20, n_gen=5, seed=SEED)


# T13 界面隔离：前端源码/页面不显示 TODO_REPLACE、initial_replaceable 等开发标记
def test_t13_ui_isolation():
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    forbidden = ["TODO_REPLACE", "initial_replaceable", "占位", "待替换", "模型版本", "model_status"]
    for token in forbidden:
        assert token not in app_src, f"app.py 不应包含开发标记: {token}"
    # 注册表元数据仅存在于后台，前端源码不得引用 model_status
    registry_src = (ROOT / "models" / "material_registry.py").read_text(encoding="utf-8")
    assert "initial_replaceable" in registry_src  # 后台存在该状态
