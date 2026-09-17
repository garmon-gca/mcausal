"""R3 — |gradient| trap under real Adam ε (not ε=0)."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcausal.schemas import ProbeReport, Status
from mcausal.update_reality import update_reality


class LinearMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Linear(16, 8, bias=False)
        nn.init.orthogonal_(self.w.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


def _one_step(c: float, seed: int, eps: float) -> dict:
    torch.manual_seed(seed)
    model = LinearMap()
    x = torch.randn(32, 16)
    y = torch.randn(32, 8)
    calib = x[:8]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=eps, betas=(0.9, 0.999))

    def loss_fn(m: nn.Module, batch: tuple) -> torch.Tensor:
        xb, yb = batch
        return c * F.mse_loss(m(xb), yb)

    return update_reality(model, opt, (x, y), calib, loss_fn=loss_fn)


def run(seed: int = 13, eps: float = 1e-8) -> ProbeReport:
    if eps == 0.0:
        raise ValueError("R3 must use real Adam epsilon, not 0")
    hi = _one_step(c=80.0, seed=seed, eps=eps)
    lo = _one_step(c=1.0, seed=seed, eps=eps)
    g_ratio = hi["gradient_norm"] / max(lo["gradient_norm"], 1e-15)
    u_ratio = hi["actual_parameter_movement"] / max(lo["actual_parameter_movement"], 1e-15)
    p_ratio = hi["optimizer_proposed_update_norm"] / max(lo["optimizer_proposed_update_norm"], 1e-15)
    f_ratio = hi["output_function_movement"]["l2"] / max(lo["output_function_movement"]["l2"], 1e-15)

    trap = (
        g_ratio >= 10.0
        and u_ratio < 0.5 * g_ratio
        and hi["levels_diverge"]
        and math.isfinite(g_ratio)
        and abs(float(hi["epsilon"]) - eps) < 1e-20
    )

    extras = {
        "eps": eps,
        "c_hi": 80.0,
        "c_lo": 1.0,
        "hi": hi,
        "lo": lo,
        "g_ratio": g_ratio,
        "update_ratio": u_ratio,
        "proposed_ratio": p_ratio,
        "function_l2_ratio": f_ratio,
        "trap": trap,
    }
    return ProbeReport(
        observed_effect=(
            f"|g| ratio={g_ratio:.3g} vs Δθ ratio={u_ratio:.3g} vs Δf ratio={f_ratio:.3g} (Adam eps={eps})"
        ),
        reproduced=True,
        current_function_match="same init (paired seed)",
        candidate_cause="Adam scale-decoupling at real ε",
        intervention="loss scale c=80 vs c=1, identical data/init",
        intervention_valid=True,
        residual="Δθ and Δf do not track |g|",
        supported=["levels_diverge", "adam_metric_trap"] if trap else [],
        rejected_or_weakened=["|g| as learning-size meter"] if trap else [],
        remaining_alternatives=["near-ε regime", "SNR change"],
        scope={"optimizer": "Adam", "eps": eps, "seed": seed},
        provenance={"case": "r3_adam_metric_trap"},
        status=Status.EFFECT_ONLY.value if trap else Status.INCONCLUSIVE.value,
        extras=extras,
    )


def passes(report: ProbeReport) -> bool:
    ex = report.extras
    if float(ex.get("eps", 0.0)) == 0.0:
        return False
    if not ex.get("trap"):
        return False
    if ex["g_ratio"] < 10:
        return False
    if ex["update_ratio"] >= 0.5 * ex["g_ratio"]:
        return False
    return True
