# -*- coding: utf-8 -*-
"""验收项 T12：基材模型可替换性 + 模块隔离标记检查。

基材公式已为正式模型（决议 R10），但"集中隔离 + 可替换"结构保留：
替换 models/substrate_models.py 内函数即可改变结果，无需改 UI / 优化器。
"""
import inspect

import numpy as np
import pytest

import models.substrate_models as sub
from models.material_registry import MATERIALS
from models.damage_chain import normalized_damage, relic_damage
from models.material_registry import get_material
from utils.spectral_io import load_spectral_data


@pytest.fixture(scope="module")
def spectral():
    return load_spectral_data()


def test_substrate_model_section_markers():
    src = inspect.getsource(sub)
    assert "SUBSTRATE_MODEL_SECTION_START" in src
    assert "SUBSTRATE_MODEL_SECTION_END" in src
    # 六种基材的 f 与 P 函数齐备
    for fn in [
        "quantity_newsprint", "quantity_bamboo_paper", "quantity_xuan_paper",
        "quantity_hemp_paper", "quantity_mulberry_paper", "quantity_silk",
        "response_newsprint", "response_bamboo_paper", "response_xuan_paper",
        "response_hemp_paper", "response_mulberry_paper", "response_silk",
    ]:
        assert fn in src, fn


def test_model_status_and_ph_metadata():
    """基材为正式模型（final）；pH 为其固定属性（仅提示，不参与计算）。"""
    substrates = [m for m in MATERIALS if m.category == "substrate"]
    pigments = [m for m in MATERIALS if m.kind == "pigment"]
    assert len(substrates) == 6
    assert len(pigments) == 12
    assert all(m.model_status == "final" for m in substrates)
    assert all(m.model_status == "final" for m in pigments)
    # pH 固定属性齐全且与文档一致
    ph = {m.name_zh: m.default_ph for m in substrates}
    assert ph == {
        "新闻纸": 5.0, "竹纸": 7.4, "宣纸": 8.5,
        "麻纸": 7.9, "桑皮纸": 8.3, "绢": 5.9,
    }


# T12 修改一个基材函数即可改变结果，无需改 app/optimizer
def test_t12_replace_one_substrate_changes_result(spectral, monkeypatch):
    mat = get_material("hemp_paper")
    wl = spectral.wavelength
    S = spectral.d55_normalized
    I, t = 60.0, 150.0

    def _chain():
        return normalized_damage(mat, wl, S, spectral.d55_normalized, I, t)

    before = _chain()

    # 仅替换 substrate_models.py 中的一个函数（改变函数形式，模拟模型更新；
    # 常数缩放会在 D_norm 比值中抵消，故需改形式）
    monkeypatch.setattr(sub, "quantity_hemp_paper",
                        lambda E_, t_: 0.5 * E_ + 0.02 * t_)
    after = _chain()

    assert not np.isclose(before[3], after[3], rtol=1e-9), "替换后结果必须改变"


def test_replace_through_relic_damage(spectral, monkeypatch):
    """通过整件损伤链验证：只动 substrate_models，整件 D_relic 变化。"""
    mats = [get_material("hemp_paper"), get_material("cochineal")]
    alpha = np.array([0.5, 0.5])
    wl = spectral.wavelength
    S = spectral.d55_normalized

    def _total():
        return relic_damage(mats, alpha, wl, S, spectral.d55_normalized, 60.0, 150.0)

    before = _total()
    monkeypatch.setattr(sub, "quantity_hemp_paper",
                        lambda E_, t_: 0.2 * E_ + 0.05 * t_)
    after = _total()
    assert not np.isclose(before, after, rtol=1e-9)
