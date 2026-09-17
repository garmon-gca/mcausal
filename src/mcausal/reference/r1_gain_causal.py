"""R1 — gain is causal for new-task speed. Standalone supervised toy.

Ground truth: function-preserving adapter/k, scale·k changes g and new-task speed;
matching g kills the residual.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcausal.history_shift import function_match_report
from mcausal.residual_match import residual_match
from mcausal.schemas import ProbeReport, Status


D_IN, H, K = 8, 8, 8
T_NEW = 12
LR = 2e-2
SEED = 7
K_GAIN = 6.0


class TinyPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(D_IN, H, bias=False)
        self.head = nn.Linear(H, K, bias=False)
        self.register_buffer("scale", torch.tensor(1.0))
        nn.init.orthogonal_(self.adapter.weight, 2.0)
        nn.init.orthogonal_(self.head.weight, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale sits on the linear adapter path so adapter/k, scale·k is function-preserving
        pre = self.scale * self.adapter(x)
        return self.head(torch.tanh(pre))


def _calib() -> torch.Tensor:
    return torch.eye(D_IN)


def _acc(model: TinyPath) -> float:
    x = _calib()
    y = torch.arange(K)
    with torch.no_grad():
        p = model(x).argmax(1)
    return float((p == y).float().mean())


def _score(model: TinyPath) -> float:
    """Mean correct-logit contrast — sensitive before accuracy saturates."""
    x = _calib()
    with torch.no_grad():
        z = model(x)
        idx = torch.arange(K)
        return float((z[idx, idx] - z.mean(dim=1)).mean())


def _gain(model: TinyPath) -> float:
    x = _calib()
    z = model(x)
    g = torch.autograd.grad(z.pow(2).mean(), model.adapter.weight)[0]
    return float(g.norm())


def _reparam(model: TinyPath, k: float) -> None:
    with torch.no_grad():
        model.adapter.weight.div_(k)
        model.scale.mul_(k)


def _train(model: TinyPath, steps: int, seed: int) -> float:
    # Only the adapter is the new-path parameter whose g we swap.
    opt = torch.optim.Adam(model.adapter.parameters(), lr=LR, eps=1e-8)
    x = _calib()
    y = torch.arange(K)
    g = torch.Generator().manual_seed(seed)
    start = _score(model)
    for _ in range(steps):
        _ = g
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return _score(model) - start


def run(seed: int = SEED) -> ProbeReport:
    torch.manual_seed(seed)
    base = TinyPath()
    low = TinyPath()
    low.load_state_dict(base.state_dict())
    high = TinyPath()
    high.load_state_dict(base.state_dict())
    _reparam(high, K_GAIN)  # adapter/k, scale·k → g rises, function stays

    fm = function_match_report(high, low, _calib(), rtol=1e-5, atol=1e-6)
    if not fm["matched"]:
        return ProbeReport(
            observed_effect="t0 function identity failed",
            reproduced=False,
            current_function_match=str(fm),
            candidate_cause="gain g",
            intervention="adapter/k, scale·k",
            intervention_valid=False,
            residual="n/a",
            status=Status.ASSAY_INVALID.value,
            extras={"function_match": fm},
        )

    g_high = _gain(high)
    g_low = _gain(low)
    # copies for effect measurement
    h1 = copy.deepcopy(high)
    l1 = copy.deepcopy(low)
    y_high = _train(h1, T_NEW, seed + 1)
    y_low = _train(l1, T_NEW, seed + 1)
    effect = y_high - y_low

    def measure_effect() -> float:
        return float(effect)

    def measure_mediator() -> dict:
        return {"g_high": g_high, "g_low": g_low, "ratio": g_high / max(g_low, 1e-12)}

    # Match: apply the same GAIN_UP reparam to LOW so g matches HIGH
    matched_low = copy.deepcopy(low)
    matched_high = copy.deepcopy(high)

    def intervene() -> dict:
        _reparam(matched_low, K_GAIN)
        fm2 = function_match_report(matched_high, matched_low, _calib(), rtol=1e-5, atol=1e-6)
        return {
            "op": "adapter/k, scale·k on LOW to match HIGH gain",
            "k": K_GAIN,
            "g_high": _gain(matched_high),
            "g_low": _gain(matched_low),
            "function_matched": fm2["matched"],
            "fm": fm2,
        }

    def invariants() -> dict:
        fm2 = function_match_report(matched_high, matched_low, _calib(), rtol=1e-5, atol=1e-6)
        gh, gl = _gain(matched_high), _gain(matched_low)
        rel = abs(gh - gl) / max(gh, 1e-12)
        return {"function_matched": fm2["matched"], "gain_rel_diff": rel, "g_high": gh, "g_low": gl}

    def invariant_ok(inv: dict) -> bool:
        return bool(inv["function_matched"]) and inv["gain_rel_diff"] < 0.05

    def effect_after() -> float:
        h2 = copy.deepcopy(matched_high)
        l2 = copy.deepcopy(matched_low)
        return _train(h2, T_NEW, seed + 1) - _train(l2, T_NEW, seed + 1)

    report = residual_match(
        suspected_variable="gain_g",
        measure_effect=measure_effect,
        measure_mediator=measure_mediator,
        intervene=intervene,
        measure_invariants=invariants,
        invariant_ok=invariant_ok,
        measure_effect_after=effect_after,
        measure_mediator_after=lambda: {
            "g_high": _gain(matched_high),
            "g_low": _gain(matched_low),
        },
        effect_threshold=0.05,
        residual_frac=0.3,
        explanations=["gain_g", "init_lottery", "hidden_history_content"],
        scope={"T_NEW": T_NEW, "K_GAIN": K_GAIN, "seed": seed, "kind": "supervised_tiny_path"},
        provenance={"case": "r1_gain_causal"},
    )
    report.extras["t0_function_match"] = fm
    report.extras["y_high"] = y_high
    report.extras["y_low"] = y_low
    report.extras["g_ratio_unmatched"] = g_high / max(g_low, 1e-12)
    return report


def passes(report: ProbeReport) -> bool:
    if report.status != Status.CAUSAL_VARIABLE_SUPPORTED.value:
        return False
    if not report.intervention_valid:
        return False
    before = abs(float(report.extras["effect_before"]))
    after = abs(float(report.extras["effect_after"]))
    if before < 0.05:
        return False
    return after <= 0.3 * before
