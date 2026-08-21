# -*- coding: utf-8 -*-
"""验收项 T12：基材模型可替换性 + 可替换接口标记检查。"""
import inspect

import numpy as np
import pytest

import models.substrate_models as sub
from models.material_registry import MATERIALS
from models.damage_chain import normalized_damage, relic_damage
from utils.spectral_io import load_spectral_data


@pytest.fixture(scope="module")
def spectral():
    return load_spectral_data()


def test_replaceable_markers_in_source():
    src = inspect.getsource(sub)
    assert "REPLACEABLE_SUBSTRATE_MODEL_START" in src
    assert "REPLACEABLE_SUBSTRATE_MODEL_END" in src
    for kw in [
        "TODO_REPLACE_SUBSTRATE_MODEL_HEMP",
        "TODO_REPLACE_SUBSTRATE_MODEL_MULBERRY",
        "TODO_REPLACE_SUBSTRATE_MODEL_NEWSPRINT",
        "TODO_REPLACE_SUBSTRATE_MODEL_XUAN",
        "TODO_REPLACE_SUBSTRATE_MODEL_BAMBOO",
    ]:
        assert kw in src, kw


def test_model_status_metadata():
    substrates = [m for m in MATERIALS if m.kind == "paper"]
    pigments = [m for m in MATERIALS if m.kind == "pigment"]
    assert len(substrates) == 5
    assert all(m.model_status == "initial_replaceable" for m in substrates)
    assert all(m.model_status == "final" for m in pigments)


# T12 修改一个基材函数即可改变结果，无需改 app/optimizer
def test_t12_replace_one_substrate_changes_result(spectral, monkeypatch):
    mat = next(m for m in MATERIALS if m.key == "hemp_paper")
    wl = spectral.wavelength
    S = spectral.d55_normalized
    I, t, pH = 60.0, 150.0, 7.5

    def _chain():
        return normalized_damage(mat, wl, S, spectral.d55_normalized, I, t, pH)

    before = _chain()

    # 仅替换 substrate_models.py 中的一个函数（改变函数形式：改指数，
    # 模拟"正式模型"；常数缩放会在 D_norm 比值中抵消，故需改形式）
    monkeypatch.setattr(sub, "quantity_hemp_paper",
                        lambda I_, t_, pH_: 0.023 * I_**0.5 * t_**0.5 * pH_**0.2)
    after = _chain()

    assert not np.isclose(before[3], after[3], rtol=1e-9), "替换后结果必须改变"


def test_replace_through_relic_damage(spectral, monkeypatch):
    """通过整件损伤链验证：只动 substrate_models，整件 D_relic 变化。"""
    from models.material_registry import get_material

    mats = [get_material("hemp_paper"), get_material("cochineal")]
    alpha = np.array([0.5, 0.5])
    wl = spectral.wavelength
    S = spectral.d55_normalized

    def _total():
        return relic_damage(mats, alpha, wl, S, spectral.d55_normalized,
                            60.0, 150.0, pH=7.5)

    before = _total()
    monkeypatch.setattr(sub, "quantity_hemp_paper",
                        lambda I_, t_, pH_: 0.023 * I_**0.5 * t_**0.5 * pH_**0.2)
    after = _total()
    assert not np.isclose(before, after, rtol=1e-9)
