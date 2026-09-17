from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mcausal.history_shift import history_shift
from mcausal.schemas import ProbeReport, Status
from mcausal.update_reality import update_reality


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.l(x)


def test_task_mismatch_assay_invalid():
    a = Tiny()
    b = Tiny()
    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=torch.eye(2),
        train_a=lambda m: {"losses": [0.0]},
        train_b=lambda m: {"losses": [0.0]},
        history_label_a="a",
        history_label_b="b",
        task_spec_a={"task": "x"},
        task_spec_b={"task": "y"},
        optimizer_config_a={},
        optimizer_config_b={},
        seed_a=0,
        seed_b=0,
    )
    assert r.status == Status.ASSAY_INVALID.value


def test_optim_mismatch_assay_invalid():
    a = Tiny()
    b = Tiny()
    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=torch.eye(2),
        train_a=lambda m: {"losses": [0.0]},
        train_b=lambda m: {"losses": [0.0]},
        history_label_a="a",
        history_label_b="a",
        task_spec_a={"task": "x"},
        task_spec_b={"task": "x"},
        optimizer_config_a={"lr": 1e-3},
        optimizer_config_b={"lr": 1e-2},
        seed_a=0,
        seed_b=0,
    )
    assert r.status == Status.ASSAY_INVALID.value


def test_loss_fn_must_be_scalar():
    m = Tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    def bad(model, batch):
        return model(batch)

    with pytest.raises(ValueError):
        update_reality(m, opt, torch.eye(2), torch.eye(2), loss_fn=bad)


def test_bad_status_rejected():
    with pytest.raises(ValueError):
        ProbeReport(
            observed_effect="x",
            reproduced=True,
            current_function_match="x",
            candidate_cause="x",
            intervention="x",
            intervention_valid=True,
            residual="x",
            status="CAUSAL_SCORE_87",
        )
