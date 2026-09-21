# -*- coding: utf-8 -*-
"""18 种材料注册表与模型版本元数据。

组成（《基材公式补充.docx》到位后）：
- 有机颜料 9 种、无机颜料 3 种（f/P 来自《公式集合_初版》）；
- 基材 6 种：新闻纸、竹纸、宣纸、麻纸、桑皮纸、绢
  （f/P 来自《基材公式补充.docx》**正式公式**，model_status="final"）。

关键口径（《需求确认记录.md》决议 R10–R14）：
- **pH 是各基材的固定属性**（公式在该 pH 下成立），不再是计算变量与界面输入项，
  仅作界面只读提示（default_ph 字段）；
- 照度符号：文档中的 E 即本项目其余部分的 I（lx）；
- f(E,t) 负值按 0 处理（在 substrate_models.py 内实现）。

可替换机制：f/P 按 (模块, 函数名) **动态解析**——替换 models/substrate_models.py
中的函数后无需修改本注册表、UI、damage_chain.py 或 optimizer.py（对应验收 T12）。
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
    kind: str                     # "pigment" / "paper" / "silk"
    f_module: ModuleType          # 包含 f 函数的模块
    f_name: str                   # f 函数名
    p_module: ModuleType          # 包含 P 函数的模块
    p_name: str                   # P 函数名
    model_status: str             # "final"（正式模型）
    version: str                  # 模型版本元数据
    default_ph: float | None = None   # 基材固定 pH（仅界面提示，不参与计算）

    @property
    def is_substrate(self) -> bool:
        return self.category == "substrate"

    def f_value(self, E, t):
        """照度—时间损伤 f(E, t)。E 为照度（lx），t 为时间（h）。"""
        fn = getattr(self.f_module, self.f_name)
        return float(fn(E, t))

    def p_value(self, wavelength: np.ndarray) -> np.ndarray:
        """光谱响应率 P(λ)。"""
        fn = getattr(self.p_module, self.p_name)
        return np.asarray(fn(wavelength), dtype=float)


def _spec(key, name_zh, category, kind, f_mod, f_name, p_mod, p_name,
          model_status, version, default_ph=None):
    return MaterialSpec(
        key=key, name_zh=name_zh, category=category, kind=kind,
        f_module=f_mod, f_name=f_name, p_module=p_mod, p_name=p_name,
        model_status=model_status, version=version, default_ph=default_ph,
    )


# ---------------------------------------------------------------------------
# 注册表构建（18 种材料）
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
    # ---- 基材（6，正式公式，pH 为固定属性） ----
    _spec("newsprint", "新闻纸", "substrate", "paper", sub, "quantity_newsprint", sub, "response_newsprint", "final", "official-v1", default_ph=sub.PH_BY_SUBSTRATE["newsprint"]),
    _spec("bamboo_paper", "竹纸", "substrate", "paper", sub, "quantity_bamboo_paper", sub, "response_bamboo_paper", "final", "official-v1", default_ph=sub.PH_BY_SUBSTRATE["bamboo_paper"]),
    _spec("xuan_paper", "宣纸", "substrate", "paper", sub, "quantity_xuan_paper", sub, "response_xuan_paper", "final", "official-v1", default_ph=sub.PH_BY_SUBSTRATE["xuan_paper"]),
    _spec("hemp_paper", "麻纸", "substrate", "paper", sub, "quantity_hemp_paper", sub, "response_hemp_paper", "final", "official-v1", default_ph=sub.PH_BY_SUBSTRATE["hemp_paper"]),
    _spec("mulberry_paper", "桑皮纸", "substrate", "paper", sub, "quantity_mulberry_paper", sub, "response_mulberry_paper", "final", "official-v1", default_ph=sub.PH_BY_SUBSTRATE["mulberry_paper"]),
    _spec("silk", "绢", "substrate", "silk", sub, "quantity_silk", sub, "response_silk", "final", "official-v1", default_ph=sub.PH_BY_SUBSTRATE["silk"]),
]

MATERIALS_BY_KEY = {m.key: m for m in MATERIALS}
MATERIALS_BY_NAME = {m.name_zh: m for m in MATERIALS}

ALL_NAMES = [m.name_zh for m in MATERIALS]
PIGMENT_NAMES = [m.name_zh for m in MATERIALS if m.kind == "pigment"]
SUBSTRATE_NAMES = [m.name_zh for m in MATERIALS if m.category == "substrate"]


def get_material(name_or_key: str) -> MaterialSpec:
    """按中文名或英文键获取材料。"""
    if name_or_key in MATERIALS_BY_NAME:
        return MATERIALS_BY_NAME[name_or_key]
    if name_or_key in MATERIALS_BY_KEY:
        return MATERIALS_BY_KEY[name_or_key]
    raise KeyError(f"未知材料: {name_or_key}")


def selected_specs(names: list[str]) -> list[MaterialSpec]:
    return [get_material(n) for n in names]


def substrate_ph_table() -> list[dict]:
    """基材固定 pH 提示表（供界面展示，不参与计算）。"""
    return [
        {"材料": m.name_zh, "pH（固定属性）": m.default_ph}
        for m in MATERIALS if m.category == "substrate"
    ]
