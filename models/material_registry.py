# -*- coding: utf-8 -*-
"""17 种材料注册表与模型版本元数据。

v3 第 5 节 / 第 8 节：
- 有机颜料 9 种、无机颜料 3 种、纸质基材 5 种；
- 五种基材后台元数据 model_status="initial_replaceable"，
  该状态不得显示在 Streamlit 前端；正式模型替换后改为 "final"；
- material_registry 负责按材料名称分派到对应 f 与 P（v3 第 6.5 节），
  优化器中不得分别编写 17 套组合逻辑。

可替换机制：f/P 按 (模块, 函数名) **动态解析**——替换 models/substrate_models.py
中的函数（含运行期替换）后无需修改本注册表、UI、damage_chain.py 或 optimizer.py，
新函数即刻生效（对应验收 T12）。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np

from models import organic_models as org
from models import inorganic_models as inorg
from models import substrate_models as sub


@dataclass(frozen=True)
class MaterialSpec:
    """单个材料的完整接口描述。"""

    key: str                      # 稳定英文键（代码内部使用）
    name_zh: str                  # 中文显示名（UI 使用）
    category: str                 # "organic" / "inorganic" / "substrate"
    kind: str                     # "pigment" / "paper"
    f_module: ModuleType          # 包含 f 函数的模块
    f_name: str                   # f 函数名
    p_module: ModuleType          # 包含 P 函数的模块
    p_name: str                   # P 函数名
    model_status: str             # "final" / "initial_replaceable"
    version: str                  # 模型版本元数据
    requires_ph: bool = False     # 是否需要 pH 输入

    def f_value(self, I, t, pH=None):
        fn = getattr(self.f_module, self.f_name)
        if self.requires_ph:
            if pH is None:
                raise ValueError(f"材料 {self.name_zh} 需要 pH 输入")
            return float(fn(I, t, pH))
        return float(fn(I, t))

    def p_value(self, wavelength: np.ndarray) -> np.ndarray:
        fn = getattr(self.p_module, self.p_name)
        return np.asarray(fn(wavelength), dtype=float)


def _spec(key, name_zh, category, kind, f_mod, f_name, p_mod, p_name,
          model_status, version, requires_ph=False):
    return MaterialSpec(
        key=key, name_zh=name_zh, category=category, kind=kind,
        f_module=f_mod, f_name=f_name, p_module=p_mod, p_name=p_name,
        model_status=model_status, version=version, requires_ph=requires_ph,
    )


# ---------------------------------------------------------------------------
# 注册表构建（17 种材料）
# ---------------------------------------------------------------------------
MATERIALS: list[MaterialSpec] = [
    # ---- 有机颜料（9） ----
    _spec("cochineal", "胭脂虫", "organic", "pigment", org, "f_cochineal", org, "P_cochineal", "final", "v1"),
    _spec("madder", "茜草", "organic", "pigment", org, "f_madder", org, "P_madder", "final", "v1"),
    _spec("shellac", "紫胶", "organic", "pigment", org, "f_shellac", org, "P_shellac", "final", "v1"),
    _spec("sappanwood", "苏木", "organic", "pigment", org, "f_sappanwood", org, "P_sappanwood", "final", "v1"),
    _spec("sophora", "槐米", "organic", "pigment", org, "f_sophora", org, "P_sophora", "final", "v1"),
    _spec("gardenia", "栀子", "organic", "pigment", org, "f_gardenia", org, "P_gardenia", "final", "v1"),
    _spec("phellodendron", "黄檗", "organic", "pigment", org, "f_phellodendron", org, "P_phellodendron", "final", "v1"),
    _spec("gamboge", "藤黄", "organic", "pigment", org, "f_gamboge", org, "P_gamboge", "final", "v1"),
    _spec("indigo", "花青", "organic", "pigment", org, "f_indigo", org, "P_indigo", "final", "v1"),
    # ---- 无机颜料（3） ----
    _spec("realgar", "雄黄", "inorganic", "pigment", inorg, "f_realgar", inorg, "P_realgar", "final", "v1"),
    _spec("red_lead", "铅丹", "inorganic", "pigment", inorg, "f_red_lead", inorg, "P_red_lead", "final", "v1"),
    _spec("cinnabar", "朱砂", "inorganic", "pigment", inorg, "f_cinnabar", inorg, "P_cinnabar", "final", "v1"),
    # ---- 纸质基材（5，初版/可替换） ----
    _spec("hemp_paper", "麻纸", "substrate", "paper", sub, "quantity_hemp_paper", sub, "response_hemp_paper", "initial_replaceable", "v1-placeholder", requires_ph=True),
    _spec("mulberry_paper", "桑纸", "substrate", "paper", sub, "quantity_mulberry_paper", sub, "response_mulberry_paper", "initial_replaceable", "v1-placeholder", requires_ph=True),
    _spec("newsprint", "新闻纸", "substrate", "paper", sub, "quantity_newsprint", sub, "response_newsprint", "initial_replaceable", "v1-placeholder", requires_ph=True),
    _spec("xuan_paper", "宣纸", "substrate", "paper", sub, "quantity_xuan_paper", sub, "response_xuan_paper", "initial_replaceable", "v1-placeholder", requires_ph=True),
    _spec("bamboo_paper", "竹纸", "substrate", "paper", sub, "quantity_bamboo_paper", sub, "response_bamboo_paper", "initial_replaceable", "v1-placeholder", requires_ph=True),
]

MATERIALS_BY_KEY = {m.key: m for m in MATERIALS}
MATERIALS_BY_NAME = {m.name_zh: m for m in MATERIALS}

ALL_NAMES = [m.name_zh for m in MATERIALS]
PIGMENT_NAMES = [m.name_zh for m in MATERIALS if m.kind == "pigment"]
SUBSTRATE_NAMES = [m.name_zh for m in MATERIALS if m.kind == "paper"]


def get_material(name_or_key: str) -> MaterialSpec:
    """按中文名或英文键获取材料。"""
    if name_or_key in MATERIALS_BY_NAME:
        return MATERIALS_BY_NAME[name_or_key]
    if name_or_key in MATERIALS_BY_KEY:
        return MATERIALS_BY_KEY[name_or_key]
    raise KeyError(f"未知材料: {name_or_key}")


def selected_specs(names: list[str]) -> list[MaterialSpec]:
    return [get_material(n) for n in names]
