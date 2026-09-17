"""ResidualMatch: effect → suspected mediator → match/intervene → residual.

Framework, not automatic magic. The user supplies measurement and intervention.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .schemas import ProbeReport, Status, jsonable


def residual_match(
    *,
    suspected_variable: str,
    measure_effect: Callable[[], float],
    measure_mediator: Callable[[], Any],
    intervene: Callable[[], dict[str, Any]],
    measure_invariants: Callable[[], dict[str, Any]],
    invariant_ok: Callable[[dict[str, Any]], bool],
    measure_effect_after: Callable[[], float],
    measure_mediator_after: Optional[Callable[[], Any]] = None,
    effect_threshold: float = 0.03,
    residual_frac: float = 0.3,
    explanations: Optional[list[str]] = None,
    scope: Optional[dict[str, Any]] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> ProbeReport:
    """Run the four-step protocol once.

    `intervene()` must perform the matching/intervention and return a dict
    describing what it did (stored in extras).
    """
    explanations = list(explanations or [])
    scope = dict(scope or {})
    provenance = dict(provenance or {"component": "ResidualMatch"})

    effect_before = float(measure_effect())
    med_before = measure_mediator()
    inv_before = measure_invariants()

    if not math_finite(effect_before):
        return ProbeReport(
            observed_effect="non-finite effect",
            reproduced=False,
            current_function_match="unknown",
            candidate_cause=suspected_variable,
            intervention="not run",
            intervention_valid=False,
            residual="n/a",
            supported=[],
            rejected_or_weakened=[],
            remaining_alternatives=explanations,
            scope=scope,
            provenance=provenance,
            status=Status.ASSAY_INVALID.value,
            extras={"reason": "non-finite effect_before"},
        )

    iv = intervene()
    inv_after = measure_invariants()
    valid = bool(invariant_ok(inv_after))
    effect_after = float(measure_effect_after())
    med_after = measure_mediator_after() if measure_mediator_after else None

    extras = {
        "effect_before": effect_before,
        "effect_after": effect_after,
        "mediator_before": jsonable(med_before),
        "mediator_after": jsonable(med_after),
        "invariants_before": jsonable(inv_before),
        "invariants_after": jsonable(inv_after),
        "intervention_record": jsonable(iv),
        "effect_threshold": effect_threshold,
        "residual_frac": residual_frac,
    }

    if not valid:
        return ProbeReport(
            observed_effect=f"effect_before={effect_before:.6g}",
            reproduced=True,
            current_function_match=str(inv_after),
            candidate_cause=suspected_variable,
            intervention=str(iv),
            intervention_valid=False,
            residual="not interpretable — invariants failed",
            supported=[],
            rejected_or_weakened=[],
            remaining_alternatives=explanations,
            scope=scope,
            provenance=provenance,
            status=Status.ASSAY_INVALID.value,
            extras=extras,
        )

    residual = effect_after
    abs_before = abs(effect_before)
    weakened = []
    supported: list[str] = []
    remaining = list(explanations)

    if abs_before < effect_threshold:
        status = Status.INCONCLUSIVE.value
        observed = f"effect_before={effect_before:.6g} below threshold {effect_threshold}"
        residual_txt = f"effect_after={effect_after:.6g} (no effect to mediate)"
    elif abs(residual) <= residual_frac * abs_before and abs(residual) < abs_before:
        status = Status.CAUSAL_VARIABLE_SUPPORTED.value
        observed = f"effect_before={effect_before:.6g}"
        residual_txt = f"effect_after={effect_after:.6g} (residual ≤ {residual_frac} of before)"
        supported = [suspected_variable]
        for e in explanations:
            if e != suspected_variable and e in remaining:
                weakened.append(e)
                remaining.remove(e)
        if suspected_variable in remaining:
            remaining.remove(suspected_variable)
    elif abs(residual) >= (1.0 - 1e-9) * abs_before:
        status = Status.CAUSAL_VARIABLE_REJECTED.value
        observed = f"effect_before={effect_before:.6g}"
        residual_txt = f"effect_after={effect_after:.6g} (matching did not move the effect)"
        weakened = [suspected_variable]
        if suspected_variable in remaining:
            remaining.remove(suspected_variable)
    else:
        status = Status.RESIDUAL_REMAINS.value
        observed = f"effect_before={effect_before:.6g}"
        residual_txt = f"effect_after={effect_after:.6g} (partial)"
        supported = [f"{suspected_variable} (partial)"]

    return ProbeReport(
        observed_effect=observed,
        reproduced=True,
        current_function_match=str(inv_after),
        candidate_cause=suspected_variable,
        intervention=str(iv),
        intervention_valid=True,
        residual=residual_txt,
        supported=supported,
        rejected_or_weakened=weakened,
        remaining_alternatives=remaining,
        scope=scope,
        provenance=provenance,
        status=status,
        extras=extras,
    )


def math_finite(x: float) -> bool:
    try:
        return x == x and abs(x) != float("inf")
    except Exception:
        return False
