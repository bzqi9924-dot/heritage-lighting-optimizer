# -*- coding: utf-8 -*-
"""Streamlit 主界面：文物照明保护—展示双目标优化工具。

布局（《产品说明 v3》第 12 节）：
    A 数据状态 → B 文物输入 → C 优化设置（高级折叠）→ D 运行 → E 结果
可增加"计算明细"折叠区查看代表方案各材料贡献（默认折叠）。
前端不得显示任何内部模型状态与开发标记（v3 第 8 / 14 / T13 节）。
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    ILLUMINANCE_MAX,
    ILLUMINANCE_MIN,
    PH_DEFAULT,
    PH_MAX,
    PH_MIN,
    TIME_DEFAULT,
    TIME_MAX,
    TIME_MIN,
)
from models.color_metrics import backend_name
from models.material_registry import ALL_NAMES, PIGMENT_NAMES, SUBSTRATE_NAMES, selected_specs
from models.optimizer import NoFeasibleSolutionError, run_optimization
from utils import export
from utils.spectral_io import DataValidationError, SpectralData, load_spectral_data

st.set_page_config(page_title="文物照明保护—展示双目标优化工具", layout="wide")

st.title("文物照明保护—展示双目标优化工具")
st.caption("基于 10 路 LED 光谱的 保护（最小化损伤 D_relic）— 展示（最大化效果 F）双目标 NSGA-II 优化")

_UPLOAD_DIR = Path(tempfile.gettempdir()) / "heritage_lighting_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TONE_OPTIONS = {"冷色调": "cold", "暖色调": "warm", "中性色调": "neutral"}


# ---------------------------------------------------------------------------
# A 数据状态
# ---------------------------------------------------------------------------
def _save_upload(uploaded) -> Path:
    p = _UPLOAD_DIR / uploaded.name
    p.write_bytes(uploaded.getbuffer())
    return p


def load_spectral_state() -> SpectralData:
    if "spectral" not in st.session_state:
        st.session_state["spectral"] = load_spectral_data()
        st.session_state["led_source"] = "data/led_10_spd.xlsx（默认）"
        st.session_state["d55_source"] = "data/d55_spd.xlsx（默认）"
    return st.session_state["spectral"]


def render_zone_a() -> SpectralData:
    st.subheader("A · 数据状态")
    led_up = st.file_uploader("上传 LED 光谱文件（覆盖默认，可选）", type=["xlsx", "csv"],
                              key="up_led")
    d55_up = st.file_uploader("上传 D55 参考光源文件（覆盖默认，可选）", type=["xlsx", "csv"],
                              key="up_d55")
    try:
        if led_up is not None or d55_up is not None:
            led_file = _save_upload(led_up) if led_up is not None else None
            d55_file = _save_upload(d55_up) if d55_up is not None else None
            spectral = load_spectral_data(led_file=led_file, d55_file=d55_file)
            st.session_state["spectral"] = spectral
            st.session_state["led_source"] = led_up.name if led_up else st.session_state.get("led_source")
            st.session_state["d55_source"] = d55_up.name if d55_up else st.session_state.get("d55_source")
            st.success("上传文件已生效（当前会话）")
        else:
            spectral = load_spectral_state()
    except DataValidationError as exc:
        st.error(f"数据校验失败：{exc}")
        spectral = load_spectral_state()

    c1, c2, c3 = st.columns(3)
    c1.metric("LED 通道数", spectral.n_channels)
    c2.metric("波长范围 (nm)", f"{spectral.wavelength[0]:.0f}–{spectral.wavelength[-1]:.0f}")
    c3.metric("步长 (nm)", f"{spectral.wavelength[1] - spectral.wavelength[0]:.1f}")
    st.caption(f"LED 来源：{st.session_state.get('led_source')} ｜ "
               f"D55 来源：{st.session_state.get('d55_source')} ｜ "
               f"颜色指标后端：{backend_name()}")
    return spectral


# ---------------------------------------------------------------------------
# B 文物输入
# ---------------------------------------------------------------------------
def render_zone_b() -> tuple[list[str], np.ndarray, float, float | None, str]:
    st.subheader("B · 文物输入")
    with st.expander("材料选择与面积比例（勾选参与计算的材料）", expanded=True):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**颜料**")
            pigments = [
                st.checkbox(n, key=f"mat_{n}") for n in PIGMENT_NAMES
            ]
        with col_r:
            st.markdown("**纸质基材**")
            substrates = [
                st.checkbox(n, key=f"mat_{n}") for n in SUBSTRATE_NAMES
            ]
        selected = [n for n, chk in zip(PIGMENT_NAMES + SUBSTRATE_NAMES, pigments + substrates) if chk]
        alpha = np.array([])
        if not selected:
            st.info("请至少勾选一种材料")
        elif len(selected) == 1:
            alpha = np.array([1.0])
        else:
            st.markdown("**面积比例 α**（可见受光表面面积占比，总和须为 1）")
            cols = st.columns(len(selected))
            alpha_vals = []
            default = 1.0 / len(selected)
            for i, (c, name) in enumerate(zip(cols, selected)):
                v = c.number_input(name, min_value=0.0, max_value=1.0,
                                   value=round(default, 4), step=0.01,
                                   key=f"alpha_{name}", format="%.4f")
                alpha_vals.append(v)
            alpha = np.array(alpha_vals)
            total = float(alpha.sum())
            if abs(total - 1.0) > 1e-4:
                st.warning(f"面积比例总和为 {total:.4f}，运行前需修正为 1")
                if st.button("自动归一化比例"):
                    alpha = alpha / total
                    for i, name in enumerate(selected):
                        st.session_state[f"alpha_{name}"] = round(float(alpha[i]), 4)
                    st.rerun()

    c1, c2, c3 = st.columns(3)
    t = c1.number_input("展示时间 t (h)", min_value=TIME_MIN, max_value=TIME_MAX,
                        value=TIME_DEFAULT, step=10.0, format="%.1f", key="input_t")
    tone_label = c2.selectbox("画作色调类型", list(TONE_OPTIONS.keys()))
    tone = TONE_OPTIONS[tone_label]

    has_substrate = any(n in SUBSTRATE_NAMES for n in selected)
    pH = None
    if has_substrate:
        pH = c3.number_input("纸质基材 pH（全画作共用）", min_value=PH_MIN, max_value=PH_MAX,
                             value=PH_DEFAULT, step=0.1, format="%.2f", key="input_ph")
    else:
        c3.caption("未选择纸质基材，无需 pH")
    return selected, alpha, t, pH, tone


# ---------------------------------------------------------------------------
# C 优化设置（高级折叠）
# ---------------------------------------------------------------------------
def render_zone_c() -> dict:
    st.subheader("C · 优化设置")
    with st.expander("高级设置（默认参数见文档第 10 节）"):
        c1, c2, c3 = st.columns(3)
        pop = c1.number_input("种群规模", min_value=20, max_value=500, value=100, step=10)
        gen = c2.number_input("最大迭代代数", min_value=10, max_value=500, value=200, step=10)
        seed = c3.number_input("随机种子", min_value=0, max_value=99999, value=42, step=1)
        c4, c5 = st.columns(2)
        sbx_p = c4.slider("交叉概率 (SBX)", 0.0, 1.0, 0.90, 0.05)
        pm_p = c5.slider("变异概率 (PM)", 0.0, 1.0, 0.10, 0.05)
    return {"pop": int(pop), "gen": int(gen), "seed": int(seed),
            "sbx": float(sbx_p), "pm": float(pm_p)}


# ---------------------------------------------------------------------------
# D 运行
# ---------------------------------------------------------------------------
def render_zone_d(spectral: SpectralData, selected, alpha, t, pH, tone, settings) -> None:
    st.subheader("D · 运行")
    run_btn = st.button("开始优化", type="primary")
    if not run_btn:
        return
    # 校验
    if not selected:
        st.error("请至少勾选一种材料")
        return
    if np.any(alpha < 0) or abs(float(np.sum(alpha)) - 1.0) > 1e-4:
        st.error(f"面积比例必须非负且总和为 1（当前 {float(np.sum(alpha)):.4f}），请修正或使用自动归一化")
        return
    if any(n in SUBSTRATE_NAMES for n in selected) and pH is None:
        st.error("选中纸质基材但缺少 pH，无法运行")
        return

    materials = selected_specs(selected)
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def _progress(cur: int, total: int) -> None:
        progress_bar.progress(min(cur / max(total, 1), 1.0))
        status_text.text(f"优化进度：第 {cur}/{total} 代")

    try:
        with st.spinner("NSGA-II 优化中（默认参数约需 1–3 分钟）……"):
            result = run_optimization(
                spectral=spectral,
                materials=materials,
                alpha=alpha,
                t=t,
                tone=tone,
                pH=pH,
                pop_size=settings["pop"],
                n_gen=settings["gen"],
                sbx_prob=settings["sbx"],
                pm_prob=settings["pm"],
                seed=settings["seed"],
                progress_callback=_progress,
            )
        st.session_state["result"] = result
        progress_bar.progress(1.0)
        status_text.text("优化完成")
        st.success(f"优化完成：可行 Pareto 解 {result.run_info['n_feasible']} 个")
    except NoFeasibleSolutionError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"运行失败：{exc}")


# ---------------------------------------------------------------------------
# E 结果
# ---------------------------------------------------------------------------
def render_zone_e(spectral: SpectralData) -> None:
    result = st.session_state.get("result")
    if result is None:
        st.info("运行优化后在此查看结果")
        return

    st.subheader("E · 结果")
    reps = result.representatives
    rep_df = result.representative_df

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**保护—展示 Pareto 前沿**")
        st.image(export.pareto_png(result.pareto_df, reps), width=640)
    with c2:
        st.markdown("**三类代表方案**")
        st.dataframe(rep_df.set_index("方案类型"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**代表方案光谱分布（含 D55 参考）**")
        spds = {name: rep.spd for name, rep in reps.items()}
        spds["D55"] = spectral.d55_normalized
        st.image(export.spd_png(spectral.wavelength, spds), width=560)
    with c4:
        st.markdown("**10 通道权重**")
        w = {name: rep.weights for name, rep in reps.items()}
        st.image(export.weights_png(w), width=560)

    # 颜色参数表
    st.markdown("**颜色参数（Rf / Rg / CCT / Duv）**")
    color_rows = []
    for name, rep in reps.items():
        color_rows.append({
            "方案类型": name,
            "Rf": round(rep.metrics.Rf, 2),
            "Rg": round(rep.metrics.Rg, 2),
            "CCT (K)": round(rep.metrics.CCT, 1),
            "Duv": round(rep.metrics.Duv, 5),
            "I (lx)": round(rep.I, 2),
            "D_relic": round(rep.D_relic, 6),
            "F": round(rep.F, 4),
        })
    st.dataframe(pd.DataFrame(color_rows), width="stretch")

    # 计算明细（默认折叠）
    with st.expander("计算明细（各材料 K / f / D_raw / D0 / D_norm / α·D_norm）"):
        for name, rep in reps.items():
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame(rep.details), width="stretch")

    # 下载
    st.markdown("**下载**")
    dl1, dl2, dl3, dl4, dl5 = st.columns(5)
    dl1.download_button("Pareto CSV", export.pareto_csv(result.pareto_df),
                        file_name="pareto_solutions.csv", mime="text/csv")
    dl2.download_button("代表方案 CSV", export.representative_csv(rep_df),
                        file_name="representative_solutions.csv", mime="text/csv")
    dl3.download_button("代表 SPD CSV", export.spd_csv(spectral.wavelength, spds),
                        file_name="representative_spd.csv", mime="text/csv")
    dl4.download_button("Pareto 前沿 PNG", export.pareto_png(result.pareto_df, reps),
                        file_name="pareto_front.png", mime="image/png")
    frames = {"Pareto 解集": result.pareto_df, "代表方案": rep_df}
    for i, (name, rep) in enumerate(reps.items()):
        frames[f"{name}明细"] = pd.DataFrame(rep.details)
    dl5.download_button("全部 XLSX", export.to_xlsx(frames),
                        file_name="optimization_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def main() -> None:
    spectral = render_zone_a()
    selected, alpha, t, pH, tone = render_zone_b()
    settings = render_zone_c()
    render_zone_d(spectral, selected, alpha, t, pH, tone, settings)
    render_zone_e(spectral)


main()
