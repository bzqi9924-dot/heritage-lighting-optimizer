# -*- coding: utf-8 -*-
"""结果导出：Pareto CSV/XLSX、代表方案 CSV/XLSX、代表 SPD CSV、
材料贡献明细、图表 PNG（matplotlib，中文字体自动注册）。

口径：《产品说明 v3》第 13 节。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# 中文字体（Windows 环境自动注册微软雅黑）
# ---------------------------------------------------------------------------
_FONT_REGISTERED = False


def register_chinese_font() -> None:
    """注册中文字体供 matplotlib 使用（跨平台）。

    依次尝试：Windows 常见字体路径 → Linux 常见 CJK 字体路径 →
    matplotlib 字体库中按名称匹配 CJK 字体。确保在 Windows 与
    Linux 服务器（Docker/云主机）上 PNG 图表中文均正常渲染。
    """
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    import matplotlib
    import matplotlib.font_manager as fm

    candidate_paths = [
        # Windows
        r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc",
        # Linux（常见 CJK 字体）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    ]
    for path in candidate_paths:
        p = Path(path)
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                name = fm.FontProperties(fname=str(p)).get_name()
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                _FONT_REGISTERED = True
                return
            except Exception:  # noqa: BLE001
                continue
    # 兜底：从 matplotlib 已扫描的字体库中按名称匹配 CJK 字体
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ("CJK", "Hei", "YaHei", "SimSun", "Song",
                                     "WenQuanYi", "Noto Sans SC", "Droid Sans Fallback")):
            matplotlib.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            _FONT_REGISTERED = True
            return
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_REGISTERED = True


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    register_chinese_font()
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# 数据导出
# ---------------------------------------------------------------------------
def pareto_csv(pareto_df: pd.DataFrame) -> bytes:
    return pareto_df.to_csv(index=False).encode("utf-8-sig")


def representative_csv(rep_df: pd.DataFrame) -> bytes:
    return rep_df.to_csv(index=False).encode("utf-8-sig")


def spd_csv(wavelength: np.ndarray, spds: dict[str, np.ndarray]) -> bytes:
    """代表 SPD CSV：第一列波长，后续列为各方案 S_ω（可含 D55）。"""
    df = pd.DataFrame({"Wavelength_nm": wavelength})
    for name, spd in spds.items():
        df[name] = np.asarray(spd)
    return df.to_csv(index=False).encode("utf-8-sig")


def details_csv(details_rows: list[dict]) -> bytes:
    df = pd.DataFrame(details_rows)
    return df.to_csv(index=False).encode("utf-8-sig")


def to_xlsx(frames: dict[str, pd.DataFrame]) -> bytes:
    """多表 XLSX。frames: {sheet名: DataFrame}。"""
    import io

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in frames.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 图表 PNG
# ---------------------------------------------------------------------------
def pareto_png(pareto_df: pd.DataFrame, reps: dict[str, object]) -> bytes:
    """Pareto 前沿图：横轴 D_relic，纵轴 F，突出三类代表方案。"""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=110)
    ax.scatter(pareto_df["D_relic"], pareto_df["F"], s=22, c="#7f9cc9",
               alpha=0.75, label="Pareto 解集")
    style = {"保护优先": ("#d62728", "o"), "展示优先": ("#2ca02c", "^"),
             "综合折中": ("#ff7f0e", "D")}
    for label, rep in reps.items():
        c, m = style.get(label, ("#333333", "o"))
        ax.scatter([rep.D_relic], [rep.F], s=90, c=c, marker=m,
                   edgecolors="k", linewidths=0.8, label=label)
    ax.set_xlabel("D_relic（保护目标，越小越好）")
    ax.set_ylabel("F（展示效果，越大越好）")
    ax.set_title("保护—展示 Pareto 前沿")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return _fig_png(fig)


def spd_png(wavelength: np.ndarray, spds: dict[str, np.ndarray]) -> bytes:
    """代表 SPD 图：三类代表方案 S_ω，叠加 D55。"""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=110)
    colors = {"保护优先": "#d62728", "展示优先": "#2ca02c",
              "综合折中": "#ff7f0e", "D55": "#555555"}
    for name, spd in spds.items():
        ax.plot(wavelength, np.asarray(spd), lw=1.6,
                label=name, color=colors.get(name, None),
                linestyle="--" if name == "D55" else "-")
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("相对光谱功率 S_ω(λ)")
    ax.set_title("三类代表方案光谱分布")
    ax.set_xlim(380, 780)
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return _fig_png(fig)


def weights_png(rep_weights: dict[str, np.ndarray]) -> bytes:
    """三类代表方案的 10 通道权重柱状图。"""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=110)
    labels = [f"LED{i+1}" for i in range(10)]
    x = np.arange(10)
    width = 0.26
    colors = {"保护优先": "#d62728", "展示优先": "#2ca02c", "综合折中": "#ff7f0e"}
    for i, (name, w) in enumerate(rep_weights.items()):
        ax.bar(x + (i - 1) * width, np.asarray(w), width, label=name,
               color=colors.get(name, None), edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("权重 ω")
    ax.set_title("三类代表方案 10 通道权重")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_png(fig)


def _fig_png(fig) -> bytes:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def save_bytes(data: bytes, filename: str, directory: Path | None = None) -> Path:
    """将字节写入 outputs/（或指定目录），返回文件路径。"""
    out = (directory or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_bytes(data)
    return path
