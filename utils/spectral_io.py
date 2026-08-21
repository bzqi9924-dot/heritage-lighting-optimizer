# -*- coding: utf-8 -*-
"""光谱数据接口：读取、校验、公共波长轴映射、最大值归一化。

口径：《产品说明 v3》第 4 节 + 《需求确认记录》决议 R8a（按位置解析，不依赖表头名称）。

- LED 文件：第 1 列为波长，后续 10 列为 10 个通道的相对光谱功率。
- D55 文件：第 1 列为波长，第 2 列为 D55 相对光谱功率。
- 公共波长轴默认 380–780 nm @1 nm；原始间隔不同时线性插值到公共轴，
  文件未完整覆盖 380–780 nm 时阻止计算并报错（不得无提示外推）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    D55_SPD_FILE,
    LED_SPD_FILE,
    NUM_LED_CHANNELS,
    WAVELENGTH_COMMON,
    WAVELENGTH_MAX,
    WAVELENGTH_MIN,
)


class DataValidationError(Exception):
    """光谱数据不满足要求时抛出（对应 v3 第 14 节错误处理）。"""


def _read_positional(path: Path, n_spd_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """按位置读取 xlsx：第 1 列=波长，随后 n_spd_cols 列=SPD。

    不依赖表头名称（决议 R8a）。兼容 sheet 名任意、表头行有无。
    """
    if not Path(path).exists():
        raise DataValidationError(f"光谱文件不存在: {path}")
    try:
        df = pd.read_excel(path, header=None)
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError(f"无法读取光谱文件 {path}: {exc}") from exc

    raw = df.to_numpy(dtype=object)
    if raw.ndim != 2 or raw.shape[1] < n_spd_cols + 1:
        raise DataValidationError(
            f"文件 {path.name} 列数不足：需要 1 列波长 + {n_spd_cols} 列 SPD，"
            f"实际 {raw.shape[1]} 列"
        )
    # 逐列宽容转换为 float（表头文字/空值 -> NaN）
    ncols = n_spd_cols + 1
    num = np.full((raw.shape[0], ncols), np.nan, dtype=float)
    for j in range(ncols):
        num[:, j] = pd.to_numeric(pd.Series(raw[:, j]), errors="coerce").to_numpy()
    # 丢弃波长为 NaN 的行（表头行 A 列为空，或表头写有文字）
    data = num[~np.isnan(num[:, 0])]
    if data.shape[0] == 0:
        raise DataValidationError(f"文件 {path.name} 未找到有效数据行")
    wavelength = data[:, 0]
    spd = data[:, 1:ncols]
    return wavelength, spd


def _validate_axis(wavelength: np.ndarray, label: str) -> None:
    if wavelength.size == 0 or not np.all(np.isfinite(wavelength)):
        raise DataValidationError(f"{label}: 波长列为空或包含非有限值")
    if wavelength.size < 2:
        raise DataValidationError(f"{label}: 波长数据点不足")
    if np.any(np.diff(wavelength) <= 0):
        raise DataValidationError(f"{label}: 波长必须严格递增")


def _check_coverage(wavelength: np.ndarray, label: str) -> None:
    """必须完整覆盖 380–780 nm，否则阻止运行（v3 第 4.3 / 14 节）。"""
    if wavelength[0] > WAVELENGTH_MIN + 1e-9 or wavelength[-1] < WAVELENGTH_MAX - 1e-9:
        raise DataValidationError(
            f"{label}: 波长范围 [{wavelength[0]:.1f}, {wavelength[-1]:.1f}] nm "
            f"未完整覆盖 {WAVELENGTH_MIN}–{WAVELENGTH_MAX} nm，无法计算。"
            f"请提供覆盖完整区间的光谱文件。"
        )


def _map_to_common(wavelength: np.ndarray, values: np.ndarray) -> np.ndarray:
    """线性插值到公共波长轴（1 nm 等间隔）。仅内插，不外推。"""
    common = np.arange(WAVELENGTH_MIN, WAVELENGTH_MAX + 1, WAVELENGTH_COMMON[2])
    return np.interp(common, wavelength, values)


def max_normalize(values: np.ndarray) -> np.ndarray:
    """最大值归一化。每列独立归一化到最大值 1（v3 第 4.3 节）。"""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mx = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(mx == 0):
        raise DataValidationError("光谱数据全为 0，无法归一化")
    return values / mx


@dataclass
class SpectralData:
    """加载并统一处理后的光谱数据。"""

    wavelength: np.ndarray                      # 公共波长轴（nm）
    led_raw: np.ndarray                         # (10, N) 原始 LED
    d55_raw: np.ndarray                         # (N,)   原始 D55
    led_normalized: np.ndarray = field(init=False)   # (10, N) 每通道最大值归一化
    d55_normalized: np.ndarray = field(init=False)   # (N,)   最大值归一化 S0
    led_labels: list[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.led_raw.shape[0] != NUM_LED_CHANNELS:
            raise DataValidationError(
                f"LED 通道数量必须为 {NUM_LED_CHANNELS}，实际 {self.led_raw.shape[0]}"
            )
        if not np.all(np.isfinite(self.led_raw)) or not np.all(np.isfinite(self.d55_raw)):
            raise DataValidationError("光谱数据包含 NaN/Inf 值")
        if np.any(np.max(self.led_raw, axis=1) == 0):
            raise DataValidationError("存在全 0 的 LED 通道")
        self.led_normalized = max_normalize(self.led_raw)
        self.d55_normalized = max_normalize(self.d55_raw).ravel()
        peaks = [448, 469, 504, 523, 565, 593, 605, 629, 665, 704]
        self.led_labels = [f"LED_{p}nm" for p in peaks]

    @property
    def n_channels(self) -> int:
        return self.led_raw.shape[0]

    @property
    def coverage_ok(self) -> bool:
        return bool(
            self.wavelength[0] <= WAVELENGTH_MIN + 1e-9
            and self.wavelength[-1] >= WAVELENGTH_MAX - 1e-9
        )


def load_spectral_data(
    led_file: str | Path | None = None, d55_file: str | Path | None = None
) -> SpectralData:
    """加载 LED 与 D55 文件，统一到公共波长轴并归一化。

    led_file / d55_file 为 None 时使用 data/ 下的默认文件（v3 第 16 节）。
    """
    led_file = Path(led_file) if led_file else LED_SPD_FILE
    d55_file = Path(d55_file) if d55_file else D55_SPD_FILE

    led_wl, led_spd = _read_positional(led_file, NUM_LED_CHANNELS)
    _validate_axis(led_wl, "LED")
    _check_coverage(led_wl, "LED")

    d55_wl, d55_spd = _read_positional(d55_file, 1)
    _validate_axis(d55_wl, "D55")
    _check_coverage(d55_wl, "D55")

    common = np.arange(WAVELENGTH_MIN, WAVELENGTH_MAX + 1, WAVELENGTH_COMMON[2])
    led_common = np.column_stack(
        [_map_to_common(led_wl, led_spd[:, q]) for q in range(NUM_LED_CHANNELS)]
    )
    d55_common = _map_to_common(d55_wl, d55_spd[:, 0])

    return SpectralData(
        wavelength=common,
        led_raw=led_common.T,
        d55_raw=d55_common,
    )


def mix_led_channels(spectral: SpectralData, weights: np.ndarray) -> np.ndarray:
    """按 10 路权重线性叠加并最大值归一化，得到候选光源 S_ω(λ)。

    v3 第 6.1 节：S_mix = Σ ω_q · S_q；S_ω = S_mix / max(S_mix)。
    权重只承担光谱形状，不承担照度标定。
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != spectral.n_channels:
        raise DataValidationError("权重个数必须等于 LED 通道数")
    s_mix = weights @ spectral.led_normalized
    mx = np.max(s_mix)
    if mx <= 0 or not np.isfinite(mx):
        raise DataValidationError("候选光源叠加结果异常（非有限或最大值≤0）")
    return s_mix / mx
