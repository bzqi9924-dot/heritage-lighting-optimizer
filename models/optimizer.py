# -*- coding: utf-8 -*-
"""NSGA-II 双目标优化：决策变量编码/修复、约束、三类代表方案。

口径：《产品说明 v3》第 6.10 / 10 / 11 节。
- 决策变量 x = [ω1…ω10, I]；实数编码；0≤ω≤1 且 Σω=1；1 lx ≤ I ≤ 300 lx
  （I 下界 1 lx 见《需求确认记录》决议 R5）；
- 目标：f1=D_relic（最小化）、f2=F（最大化）；算法内部 G=[D_relic, -F]；
- 颜色强约束（Rf≥70, Rg≥88, 2650K≤CCT≤5550K, |Duv|≤0.0054）作为显式约束
  交给优化器，不满足者为不可行解，不得进入最终 Pareto 解集；
- 三类代表方案：保护优先 argmin D、展示优先 argmax F、综合折中（最小最大
  归一化偏差：min_xk max[(D(xk)-Dmin)/(Dmax-Dmin), (Fmax-F(xk))/(Fmax-Fmin)]，
  即保护、展示两目标中表现较差者的归一化偏差尽量小，含零范围保护）。
  （决议 R9，替代 v3 第 11 节的欧氏距离规则）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from config import (
    ILLUMINANCE_MAX,
    ILLUMINANCE_MIN,
    POP_SIZE,
    N_GEN,
    PM_ETA,
    PM_PROB,
    SBX_ETA,
    SBX_PROB,
    SEED,
)
from models.color_metrics import ColorMetricsError, ColorMetricsResult, compute_color_metrics
from models.damage_chain import (
    ModelCalculationError,
    material_detail,
    relic_damage,
)
from models.display_model import display_score
from models.material_registry import MaterialSpec
from utils.spectral_io import SpectralData, mix_led_channels


class NoFeasibleSolutionError(Exception):
    """当前强约束下无可行 Pareto 解（v3 第 14 节）。"""


# ---------------------------------------------------------------------------
# 编码与修复
# ---------------------------------------------------------------------------
class WeightSampling(Sampling):
    """初始种群：ω 从单纯形均匀采样（Dirichlet），I 均匀采样。"""

    def __init__(self, seed: int = SEED):
        super().__init__()
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(self.seed)
        weights = rng.dirichlet(np.ones(problem.n_var - 1), size=n_samples)
        illum = rng.uniform(ILLUMINANCE_MIN, ILLUMINANCE_MAX, size=(n_samples, 1))
        return np.hstack([weights, illum])


class WeightRepair(Repair):
    """交叉/变异后修复：权重裁剪到非负 → sum-归一化；和接近 0 则重置为等权；
    照度裁剪到 [1, 300]（v3 第 10 节）。"""

    def _do(self, problem, X, **kwargs):
        W = np.clip(X[:, : problem.n_var - 1], 0.0, None)
        sums = W.sum(axis=1, keepdims=True)
        bad = sums[:, 0] < 1e-6
        if np.any(bad):
            W[bad] = 1.0 / (problem.n_var - 1)
            sums[bad] = 1.0
        X[:, : problem.n_var - 1] = W / sums
        X[:, problem.n_var - 1] = np.clip(
            X[:, problem.n_var - 1], ILLUMINANCE_MIN, ILLUMINANCE_MAX
        )
        return X


def decode(x: np.ndarray) -> tuple[np.ndarray, float]:
    """解码个体 -> (ω1…ω10, I)。"""
    return np.asarray(x[:10], dtype=float), float(x[10])


# ---------------------------------------------------------------------------
# 优化问题
# ---------------------------------------------------------------------------
class HeritageLightingProblem(Problem):
    """双目标：min D_relic，min -F；6 条颜色强约束。"""

    def __init__(
        self,
        spectral: SpectralData,
        materials: list[MaterialSpec],
        alpha: np.ndarray,
        t: float,
        tone: str,
        pH: float | None = None,
    ):
        self.spectral = spectral
        self.materials = materials
        self.alpha = np.asarray(alpha, dtype=float)
        self.t = float(t)
        self.tone = tone
        self.pH = pH
        super().__init__(
            n_var=11,
            n_obj=2,
            n_ieq_constr=6,
            xl=np.array([0.0] * 10 + [ILLUMINANCE_MIN]),
            xu=np.array([1.0] * 10 + [ILLUMINANCE_MAX]),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = np.empty(X.shape[0])
        f2 = np.empty(X.shape[0])
        g = np.empty((X.shape[0], self.n_ieq_constr))
        for i, x in enumerate(X):
            weights, I = decode(x)
            S = mix_led_channels(self.spectral, weights)
            try:
                cm: ColorMetricsResult = compute_color_metrics(
                    self.spectral.wavelength, S
                )
            except ColorMetricsError:
                # 单个个体颜色指标异常：标为强不可行，不进入结果（v3 第 14 节）
                g[i] = np.full(6, 1e6)
                f1[i], f2[i] = 1e9, -1e9
                continue
            g[i] = [
                70.0 - cm.Rf,
                88.0 - cm.Rg,
                2650.0 - cm.CCT,
                cm.CCT - 5550.0,
                cm.Duv - 0.0054,
                -0.0054 - cm.Duv,
            ]
            d_relic = relic_damage(
                self.materials,
                self.alpha,
                self.spectral.wavelength,
                S,
                self.spectral.d55_normalized,
                I,
                self.t,
                self.pH,
            )
            F = display_score(self.tone, I, cm.Rf, cm.Rg)
            f1[i] = d_relic
            f2[i] = F
        out["F"] = np.column_stack([f1, -f2])  # 内部统一最小化
        out["G"] = g


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class RepresentativeSolution:
    label: str                              # 保护优先 / 展示优先 / 综合折中
    weights: np.ndarray
    I: float
    D_relic: float
    F: float
    metrics: ColorMetricsResult
    spd: np.ndarray                         # S_ω
    details: list[dict]                     # 材料贡献明细


@dataclass
class OptimizationResult:
    pareto_df: object                       # pandas DataFrame
    representatives: dict[str, RepresentativeSolution]
    run_info: dict = field(default_factory=dict)

    @property
    def representative_df(self) -> object:
        import pandas as pd

        rows = []
        for rep in self.representatives.values():
            rows.append(
                {
                    "方案类型": rep.label,
                    **{f"ω{q+1}": rep.weights[q] for q in range(10)},
                    "I_lx": rep.I,
                    "D_relic": rep.D_relic,
                    "F": rep.F,
                    "Rf": rep.metrics.Rf,
                    "Rg": rep.metrics.Rg,
                    "CCT_K": rep.metrics.CCT,
                    "Duv": rep.metrics.Duv,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 代表方案提取（v3 第 11 节；综合折中规则按决议 R9 改为最小最大归一化偏差）
# ---------------------------------------------------------------------------
def _extract_representatives(
    spectral: SpectralData,
    materials: list[MaterialSpec],
    alpha: np.ndarray,
    t: float,
    tone: str,
    pH: float | None,
    X: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    cm_list: list[ColorMetricsResult],
    spd_list: list[np.ndarray],
) -> dict[str, RepresentativeSolution]:
    """从可行 Pareto 解集中提取三类代表方案（决议 R9）。

    综合折中 = 最小最大归一化偏差（minimax）：
        dev_D(x) = (D(x) - D_min) / (D_max - D_min)
        dev_F(x) = (F_max - F(x)) / (F_max - F_min)
        x_B      = argmin_x max{dev_D(x), dev_F(x)}
    即让保护、展示两目标中表现较差的那个归一化偏差尽量小。
    零范围保护：某目标在 Pareto 集中 max=min 时该目标归一化偏差记 0（避免除零）。
    """
    i_prot = int(np.argmin(D))
    i_disp = int(np.argmax(F))
    d_min, d_max = float(np.min(D)), float(np.max(D))
    f_min, f_max = float(np.min(F)), float(np.max(F))
    d_norm = (D - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(D)
    f_norm = (F - f_min) / (f_max - f_min) if f_max > f_min else np.zeros_like(F)
    deviation = np.maximum(d_norm, 1.0 - f_norm)  # 1-f_norm = (Fmax-F)/(Fmax-Fmin)
    i_comp = int(np.argmin(deviation))

    reps: dict[str, RepresentativeSolution] = {}
    for label, idx in [
        ("保护优先", i_prot),
        ("展示优先", i_disp),
        ("综合折中", i_comp),
    ]:
        weights, I = decode(X[idx])
        S = spd_list[idx]
        details = [
            material_detail(
                m,
                spectral.wavelength,
                S,
                spectral.d55_normalized,
                I,
                t,
                pH,
                float(a),
            )
            for m, a in zip(materials, alpha)
        ]
        reps[label] = RepresentativeSolution(
            label=label,
            weights=weights,
            I=I,
            D_relic=float(D[idx]),
            F=float(F[idx]),
            metrics=cm_list[idx],
            spd=S,
            details=details,
        )
    return reps


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_optimization(
    spectral: SpectralData,
    materials: list[MaterialSpec],
    alpha: np.ndarray,
    t: float,
    tone: str,
    pH: float | None = None,
    pop_size: int = POP_SIZE,
    n_gen: int = N_GEN,
    sbx_prob: float = SBX_PROB,
    pm_prob: float = PM_PROB,
    seed: int = SEED,
    progress_callback: Callable[[int, int], None] | None = None,
) -> OptimizationResult:
    """运行 NSGA-II 并返回可行 Pareto 解集与三类代表方案。"""
    alpha = np.asarray(alpha, dtype=float)
    if abs(float(np.sum(alpha)) - 1.0) > 1e-6 or np.any(alpha < 0):
        raise ValueError("面积比例必须非负且总和为 1")

    problem = HeritageLightingProblem(spectral, materials, alpha, t, tone, pH)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=WeightSampling(seed=seed),
        crossover=SBX(prob=sbx_prob, eta=SBX_ETA),
        mutation=PM(prob=pm_prob, eta=PM_ETA),
        repair=WeightRepair(),
        eliminate_duplicates=True,
    )

    class _Progress:
        def __init__(self, cb, total):
            self.cb = cb
            self.total = total

        def __call__(self, algorithm):
            if self.cb is not None:
                self.cb(int(algorithm.n_gen), self.total)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
        callback=_Progress(progress_callback, n_gen),
    )

    X, F_mat = res.X, res.F
    G = res.CV if res.CV is not None else None

    if X is None or F_mat is None:
        raise NoFeasibleSolutionError(
            "当前强约束下未获得可行解（Rf/Rg/CCT/Duv 约束过严或材料组合无解）"
        )

    # CV 为 0 且约束全部满足才算可行（v3：不可行解不得进入最终 Pareto 解集）
    feasible = np.ones(X.shape[0], dtype=bool)
    if G is not None:
        if G.ndim == 2 and G.shape[1] == 1:
            feasible = G[:, 0] <= 1e-6
        else:
            feasible = np.all(G <= 1e-6, axis=1)

    if not np.any(feasible):
        raise NoFeasibleSolutionError(
            "当前强约束下未获得可行解（Rf/Rg/CCT/Duv 约束过严或材料组合无解）"
        )

    X_f, F_f = X[feasible], F_mat[feasible]
    D = F_f[:, 0]
    F_display = -F_f[:, 1]

    # 重新计算可行个体的颜色指标与 SPD（保证与代表方案输出一致）
    cm_list: list[ColorMetricsResult] = []
    spd_list: list[np.ndarray] = []
    for x in X_f:
        weights, _ = decode(x)
        S = mix_led_channels(spectral, weights)
        spd_list.append(S)
        cm_list.append(compute_color_metrics(spectral.wavelength, S))

    representatives = _extract_representatives(
        spectral, materials, alpha, t, tone, pH,
        X_f, D, F_display, cm_list, spd_list,
    )

    import pandas as pd

    pareto_rows = []
    for i, x in enumerate(X_f):
        weights, I = decode(x)
        pareto_rows.append(
            {
                "solution_id": i,
                **{f"ω{q+1}": weights[q] for q in range(10)},
                "I_lx": I,
                "D_relic": float(D[i]),
                "F": float(F_display[i]),
                "Rf": cm_list[i].Rf,
                "Rg": cm_list[i].Rg,
                "CCT_K": cm_list[i].CCT,
                "Duv": cm_list[i].Duv,
            }
        )
    pareto_df = pd.DataFrame(pareto_rows)

    run_info = {
        "n_evaluations": int(res.exec_time if hasattr(res, "exec_time") else 0),
        "n_feasible": int(feasible.sum()),
        "pop_size": pop_size,
        "n_gen": n_gen,
        "seed": seed,
    }
    return OptimizationResult(pareto_df=pareto_df, representatives=representatives, run_info=run_info)
