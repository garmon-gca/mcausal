"""R2 — construction-null: labels differ, realized trajectory does not.

If mcausal invents a mechanism here, the suite MUST fail.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcausal.history_shift import history_shift
from mcausal.residual_match import residual_match
from mcausal.schemas import ProbeReport, Status


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.net.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 4, generator=g)
    y = torch.randn(8, 4, generator=g)
    return x, y


def _train_factory(shared_x: torch.Tensor, shared_y: torch.Tensor, steps: int = 5):
    def train(model: Tiny) -> dict:
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(shared_x), shared_y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        w = model.net.weight.detach().cpu().flatten().tolist()
        return {"losses": losses, "weight": w}

    return train


def run(seed: int = 11) -> ProbeReport:
    torch.manual_seed(seed)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x, y = _batch(seed + 3)
    calib = x[:4]
    train = _train_factory(x, y, steps=6)
    task = {"name": "mse_linear", "steps": 6}
    opt_cfg = {"name": "SGD", "lr": 0.05}

    hs = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=calib,
        train_a=train,
        train_b=train,
        history_label_a="CAUSAL_OWN_LOOP",
        history_label_b="YOKED_REPLAY",
        task_spec_a=task,
        task_spec_b=task,
        optimizer_config_a=opt_cfg,
        optimizer_config_b=opt_cfg,
        seed_a=seed,
        seed_b=seed,
        trajectory_atol=1e-8,
    )

    # ResidualMatch must also refuse a deep conclusion (no realized contrast).
    def measure_effect() -> float:
        # identical systems → effect 0
        return 0.0

    def intervene() -> dict:
        return {"op": "none — no contrast to match"}

    rm = residual_match(
        suspected_variable="causal_vs_yoked_label",
        measure_effect=measure_effect,
        measure_mediator=lambda: {"label_a": "CAUSAL_OWN_LOOP", "label_b": "YOKED_REPLAY"},
        intervene=intervene,
        measure_invariants=lambda: {"traj_identical": True},
        invariant_ok=lambda inv: True,
        measure_effect_after=lambda: 0.0,
        effect_threshold=0.03,
        explanations=["causal_vs_yoked_label"],
        scope={"note": "construction_null"},
        provenance={"case": "r2_construction_null"},
    )

    hs.extras["residual_match_status"] = rm.status
    hs.extras["residual_match"] = rm.to_dict()
    # If ResidualMatch claims the label is causal, fail the case at passes().
    hs.extras["false_causal"] = rm.status == Status.CAUSAL_VARIABLE_SUPPORTED.value
    return hs


def passes(report: ProbeReport) -> bool:
    if report.status != Status.CONSTRUCTION_NULL.value:
        return False
    if report.extras.get("false_causal"):
        return False
    rm_status = report.extras.get("residual_match_status")
    if rm_status == Status.CAUSAL_VARIABLE_SUPPORTED.value:
        return False
    # must not claim a history effect
    if report.status == Status.HISTORY_EFFECT_PRESENT.value:
        return False
    return "CONSTRUCTION_NULL" in report.supported
