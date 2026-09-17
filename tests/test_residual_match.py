from __future__ import annotations

from mcausal.residual_match import residual_match
from mcausal.schemas import Status


def test_supported_when_residual_collapses():
    state = {"x": 1.0}

    def intervene():
        state["x"] = 0.0
        return {"set": 0}

    r = residual_match(
        suspected_variable="m",
        measure_effect=lambda: 1.0,
        measure_mediator=lambda: state["x"],
        intervene=intervene,
        measure_invariants=lambda: {"ok": True},
        invariant_ok=lambda inv: inv["ok"],
        measure_effect_after=lambda: 0.01,
        effect_threshold=0.03,
        residual_frac=0.3,
        explanations=["m", "other"],
    )
    assert r.status == Status.CAUSAL_VARIABLE_SUPPORTED.value
    assert "m" in r.supported


def test_rejected_when_residual_unchanged():
    r = residual_match(
        suspected_variable="m",
        measure_effect=lambda: 1.0,
        measure_mediator=lambda: 0,
        intervene=lambda: {"noop": True},
        measure_invariants=lambda: {"ok": True},
        invariant_ok=lambda inv: True,
        measure_effect_after=lambda: 1.0,
        explanations=["m"],
    )
    assert r.status == Status.CAUSAL_VARIABLE_REJECTED.value


def test_invalid_invariants():
    r = residual_match(
        suspected_variable="m",
        measure_effect=lambda: 1.0,
        measure_mediator=lambda: 0,
        intervene=lambda: {},
        measure_invariants=lambda: {"ok": False},
        invariant_ok=lambda inv: inv["ok"],
        measure_effect_after=lambda: 0.0,
    )
    assert r.status == Status.ASSAY_INVALID.value
    assert r.intervention_valid is False
