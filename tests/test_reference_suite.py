from __future__ import annotations

from mcausal.reference import r1_gain_causal, r2_construction_null, r3_adam_metric_trap
from mcausal.schemas import Status


def test_r1_pass():
    r = r1_gain_causal.run(seed=7)
    assert r1_gain_causal.passes(r), (r.status, r.extras.get("effect_before"), r.extras.get("effect_after"))
    assert r.status == Status.CAUSAL_VARIABLE_SUPPORTED.value


def test_r2_pass_no_false_mechanism():
    r = r2_construction_null.run(seed=11)
    assert r2_construction_null.passes(r), r.status
    assert r.status == Status.CONSTRUCTION_NULL.value


def test_r2_deterministic_replay():
    a = r2_construction_null.run(seed=11).to_dict()
    b = r2_construction_null.run(seed=11).to_dict()
    assert a["status"] == b["status"]
    assert a["extras"]["traj_a"] == b["extras"]["traj_a"]
    assert a["extras"]["traj_b"] == b["extras"]["traj_b"]


def test_r3_pass_real_eps():
    r = r3_adam_metric_trap.run(seed=13, eps=1e-8)
    assert r3_adam_metric_trap.passes(r), r.extras
    assert r.extras["eps"] != 0.0


def test_r3_rejects_eps_zero():
    try:
        r3_adam_metric_trap.run(seed=13, eps=0.0)
        raise AssertionError("should have refused eps=0")
    except ValueError:
        pass
