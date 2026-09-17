"""HistoryShift: different *history*, same subsequent training protocol.

If the new-task protocol itself differs (trainable set, clip, lr groups),
that is UPDATE_ALLOCATION_EFFECT — not a history effect. Declare
`subsequent_protocol_a/b`. ResidualMatch names the mediator
(e.g. TRAINABLE_SET_MEDIATION).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .schemas import ProbeReport, Status, jsonable
from .update_reality import _outputs_to_tensor, call_model


def function_match_report(
    model_a: nn.Module,
    model_b: nn.Module,
    calibration_inputs: Any,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        ya = _outputs_to_tensor(call_model(model_a, calibration_inputs)).detach().float()
        yb = _outputs_to_tensor(call_model(model_b, calibration_inputs)).detach().float()
    diff = (ya - yb).abs()
    max_abs = float(diff.max())
    rms = float((ya - yb).pow(2).mean().sqrt())
    matched = bool(torch.allclose(ya, yb, rtol=rtol, atol=atol))
    return {
        "matched": matched,
        "max_abs": max_abs,
        "rms": rms,
        "rtol": rtol,
        "atol": atol,
        "shape": list(ya.shape),
    }


def _traj_close(ta: dict[str, Any], tb: dict[str, Any], atol: float) -> tuple[bool, float]:
    keys = sorted(set(ta) | set(tb))
    worst = 0.0
    for k in keys:
        if k not in ta or k not in tb:
            return False, float("inf")
        va, vb = ta[k], tb[k]
        if torch.is_tensor(va):
            va = va.detach().cpu().flatten().float()
            vb = torch.as_tensor(vb).detach().cpu().flatten().float()
            d = float((va - vb).abs().max()) if va.numel() == vb.numel() else float("inf")
        elif isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
            if len(va) != len(vb):
                return False, float("inf")
            d = 0.0
            for x, y in zip(va, vb):
                d = max(d, abs(float(x) - float(y)))
        else:
            try:
                d = abs(float(va) - float(vb))
            except (TypeError, ValueError):
                if va != vb:
                    return False, float("inf")
                d = 0.0
        worst = max(worst, d)
    return worst <= atol, worst


def history_shift(
    *,
    model_a: nn.Module,
    model_b: nn.Module,
    calibration_inputs: Any,
    train_a: Callable[[nn.Module], dict[str, Any]],
    train_b: Callable[[nn.Module], dict[str, Any]],
    history_label_a: str,
    history_label_b: str,
    task_spec_a: dict[str, Any],
    task_spec_b: dict[str, Any],
    optimizer_config_a: dict[str, Any],
    optimizer_config_b: dict[str, Any],
    seed_a: int,
    seed_b: int,
    function_rtol: float = 1e-5,
    function_atol: float = 1e-6,
    trajectory_atol: float = 1e-7,
    effect_atol: float = 1e-4,
    require_same_task: bool = True,
    require_same_optim: bool = True,
    skip_function_check: bool = False,
    subsequent_protocol_a: dict[str, Any] | None = None,
    subsequent_protocol_b: dict[str, Any] | None = None,
) -> ProbeReport:
    """Compare subsequent learning after two training histories.

    HistoryShift is valid only when the *following* protocol is the same
    (same trainable set, clip, lr groups, data). Different history, same recipe.

    If `subsequent_protocol_a != subsequent_protocol_b`, a detected gap is
    UPDATE_ALLOCATION_EFFECT, not HISTORY_EFFECT_PRESENT.

    Status: HISTORY_EFFECT_PRESENT | NO_DETECTED_HISTORY_EFFECT |
    CURRENT_FUNCTION_NOT_MATCHED | ASSAY_INVALID | CONSTRUCTION_NULL |
    UPDATE_ALLOCATION_EFFECT
    """
    provenance = {"component": "HistoryShift", "label_a": history_label_a, "label_b": history_label_b}
    proto_declared = subsequent_protocol_a is not None or subsequent_protocol_b is not None
    if proto_declared and (subsequent_protocol_a is None or subsequent_protocol_b is None):
        return ProbeReport(
            observed_effect="subsequent_protocol declared on only one arm",
            reproduced=False,
            current_function_match="not evaluated",
            candidate_cause="n/a",
            intervention="none",
            intervention_valid=False,
            residual="n/a",
            supported=[],
            rejected_or_weakened=["HISTORY_EFFECT_PRESENT"],
            remaining_alternatives=["declare both subsequent_protocol_a and _b"],
            scope={"seed_a": seed_a, "seed_b": seed_b},
            provenance=provenance,
            status=Status.ASSAY_INVALID.value,
            extras={"reason": "partial subsequent_protocol"},
        )
    allocation = bool(
        subsequent_protocol_a is not None
        and subsequent_protocol_b is not None
        and subsequent_protocol_a != subsequent_protocol_b
    )

    if require_same_task and task_spec_a != task_spec_b:
        return ProbeReport(
            observed_effect="new-task spec mismatch",
            reproduced=False,
            current_function_match="not evaluated",
            candidate_cause="n/a",
            intervention="none",
            intervention_valid=False,
            residual="n/a",
            supported=[],
            rejected_or_weakened=["any history-effect claim on this pair"],
            remaining_alternatives=["fix task_spec"],
            scope={"seed_a": seed_a, "seed_b": seed_b},
            provenance=provenance,
            status=Status.ASSAY_INVALID.value,
            extras={"reason": "task_spec_a != task_spec_b"},
        )
    if require_same_optim and optimizer_config_a != optimizer_config_b:
        return ProbeReport(
            observed_effect="optimizer/config mismatch",
            reproduced=False,
            current_function_match="not evaluated",
            candidate_cause="n/a",
            intervention="none",
            intervention_valid=False,
            residual="n/a",
            supported=[],
            rejected_or_weakened=["any history-effect claim on this pair"],
            remaining_alternatives=["declare require_same_optim=False if optimizer-state IS the history"],
            scope={"seed_a": seed_a, "seed_b": seed_b},
            provenance=provenance,
            status=Status.ASSAY_INVALID.value,
            extras={"reason": "optimizer_config_a != optimizer_config_b"},
        )

    if skip_function_check:
        fm = {"matched": None, "skipped": True}
        matched = True
    else:
        fm = function_match_report(model_a, model_b, calibration_inputs, rtol=function_rtol, atol=function_atol)
        matched = bool(fm["matched"])
        if not matched:
            return ProbeReport(
                observed_effect="current functions differ; subsequent learning not comparable as history effect",
                reproduced=False,
                current_function_match=f"NOT_MATCHED max_abs={fm['max_abs']:.3e} rms={fm['rms']:.3e}",
                candidate_cause="n/a — assay blocked",
                intervention="none",
                intervention_valid=False,
                residual="n/a",
                supported=[],
                rejected_or_weakened=["HISTORY_EFFECT_PRESENT (not testable)"],
                remaining_alternatives=["match current function first"],
                scope={"seed_a": seed_a, "seed_b": seed_b},
                provenance=provenance,
                status=Status.CURRENT_FUNCTION_NOT_MATCHED.value,
                extras={"function_match": fm},
            )

    traj_a = train_a(model_a)
    traj_b = train_b(model_b)
    close, worst = _traj_close(traj_a, traj_b, trajectory_atol)
    labels_differ = history_label_a != history_label_b

    extras = {
        "function_match": fm,
        "trajectory_worst_abs": worst,
        "trajectory_close": close,
        "traj_a": jsonable(traj_a),
        "traj_b": jsonable(traj_b),
        "labels_differ": labels_differ,
        "allocation": allocation,
        "subsequent_protocol_a": jsonable(subsequent_protocol_a),
        "subsequent_protocol_b": jsonable(subsequent_protocol_b),
    }

    if allocation:
        if close or worst <= effect_atol:
            return ProbeReport(
                observed_effect="subsequent protocols differ but trajectories match — no allocation contrast realized",
                reproduced=True,
                current_function_match="MATCHED" if matched else "skipped",
                candidate_cause="none realized",
                intervention="none",
                intervention_valid=False,
                residual="n/a",
                supported=[],
                rejected_or_weakened=["HISTORY_EFFECT_PRESENT (protocol was not a history contrast)"],
                remaining_alternatives=["underpowered T"],
                scope={"seed_a": seed_a, "seed_b": seed_b},
                provenance=provenance,
                status=Status.INCONCLUSIVE.value,
                extras=extras,
            )
        return ProbeReport(
            observed_effect="subsequent-learning trajectories differ because the new-task protocol differs (not a history-only contrast)",
            reproduced=True,
            current_function_match="MATCHED" if matched else "skipped",
            candidate_cause="update allocation / subsequent protocol (not named)",
            intervention="none (detection only)",
            intervention_valid=False,
            residual="use ResidualMatch; if trainable sets differ, class is TRAINABLE_SET_MEDIATION",
            supported=["UPDATE_ALLOCATION_EFFECT"],
            rejected_or_weakened=["HISTORY_EFFECT_PRESENT", "current-eval-is-sufficient"],
            remaining_alternatives=["trainable_set", "clip", "lr_groups", "weight_decay"],
            scope={"seed_a": seed_a, "seed_b": seed_b, "trajectory_worst_abs": worst},
            provenance=provenance,
            status=Status.UPDATE_ALLOCATION_EFFECT.value,
            extras=extras,
        )

    # Construction-null: different story labels, identical realized trajectory.
    if labels_differ and close:
        return ProbeReport(
            observed_effect="labels differ but realized trajectory is identical — no causal contrast was instantiated",
            reproduced=True,
            current_function_match="MATCHED" if matched else "skipped",
            candidate_cause="none realized",
            intervention="none",
            intervention_valid=False,
            residual="n/a — deep causal conclusion unavailable",
            supported=["CONSTRUCTION_NULL"],
            rejected_or_weakened=["any mechanism inferred from these labels"],
            remaining_alternatives=["build a lived difference before causal talk"],
            scope={"seed_a": seed_a, "seed_b": seed_b, "trajectory_atol": trajectory_atol},
            provenance=provenance,
            status=Status.CONSTRUCTION_NULL.value,
            extras=extras,
        )

    if close or worst <= effect_atol:
        return ProbeReport(
            observed_effect="no detected difference in subsequent-learning trajectories",
            reproduced=True,
            current_function_match="MATCHED" if matched else "skipped",
            candidate_cause="none detected",
            intervention="none",
            intervention_valid=False,
            residual="n/a",
            supported=[],
            rejected_or_weakened=["HISTORY_EFFECT_PRESENT on this assay"],
            remaining_alternatives=["underpowered T", "metric too coarse"],
            scope={"seed_a": seed_a, "seed_b": seed_b},
            provenance=provenance,
            status=Status.NO_DETECTED_HISTORY_EFFECT.value,
            extras=extras,
        )

    return ProbeReport(
        observed_effect="subsequent-learning trajectories differ after matched current function",
        reproduced=True,
        current_function_match="MATCHED" if matched else "skipped",
        candidate_cause="unspecified training history (not named)",
        intervention="none (detection only)",
        intervention_valid=False,
        residual="cause not identified — use ResidualMatch",
        supported=["HISTORY_EFFECT_PRESENT"],
        rejected_or_weakened=["current-eval-is-sufficient"],
        remaining_alternatives=["optimizer state", "data order", "unmeasured parameter"],
        scope={"seed_a": seed_a, "seed_b": seed_b, "trajectory_worst_abs": worst},
        provenance=provenance,
        status=Status.HISTORY_EFFECT_PRESENT.value,
        extras=extras,
    )
