from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcausal.history_shift import history_shift
from mcausal.schemas import Status


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        return self.l(x)


def test_construction_null_identical_traj():
    torch.manual_seed(0)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(6, 3)
    y = torch.randn(6, 3)

    def train(m):
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        losses = []
        for _ in range(3):
            opt.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return {"losses": losses, "w": m.l.weight.detach().cpu().flatten().tolist()}

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=x[:2],
        train_a=train,
        train_b=train,
        history_label_a="A_STORY",
        history_label_b="B_STORY",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={"sgd": 0.1},
        optimizer_config_b={"sgd": 0.1},
        seed_a=0,
        seed_b=0,
    )
    assert r.status == Status.CONSTRUCTION_NULL.value


def test_function_match_is_calibration_not_global():
    torch.manual_seed(1)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(6, 3)
    y = torch.randn(6, 3)

    def train(m):
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        losses = []
        for _ in range(2):
            opt.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return {"losses": losses, "w": m.l.weight.detach().cpu().flatten().tolist()}

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=x[:2],
        train_a=train,
        train_b=train,
        history_label_a="A",
        history_label_b="B",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={"sgd": 0.1},
        optimizer_config_b={"sgd": 0.1},
        seed_a=1,
        seed_b=1,
    )
    assert "supplied calibration inputs" in r.current_function_match
    assert r.extras["function_match"]["note"].startswith("empirical match")
    assert r.extras["function_match"]["calibration"]["n_rows"] == 2


def test_function_not_matched_blocks():
    torch.manual_seed(1)
    a = Tiny()
    b = Tiny()
    with torch.no_grad():
        b.l.weight.add_(1.0)

    def train(m):
        return {"losses": [0.0]}

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=torch.eye(3),
        train_a=train,
        train_b=train,
        history_label_a="a",
        history_label_b="b",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={},
        optimizer_config_b={},
        seed_a=1,
        seed_b=1,
    )
    assert r.status == Status.CURRENT_FUNCTION_NOT_MATCHED.value


def test_history_effect_when_traj_differs():
    torch.manual_seed(2)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(6, 3)
    y = torch.randn(6, 3)

    def train_fast(m):
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        losses = []
        for _ in range(4):
            opt.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return {"losses": losses}

    def train_slow(m):
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        losses = []
        for _ in range(4):
            opt.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return {"losses": losses}

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=x[:2],
        train_a=train_fast,
        train_b=train_slow,
        history_label_a="fast",
        history_label_b="slow",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={"lr": "declared_same"},
        optimizer_config_b={"lr": "declared_same"},
        seed_a=2,
        seed_b=2,
        trajectory_atol=1e-8,
    )
    assert r.status == Status.HISTORY_EFFECT_CANDIDATE.value
    assert r.extras["protocol_kind"] == "DECLARED_PROTOCOL"
    assert Status.HISTORY_EFFECT_PRESENT.value not in r.supported


def test_allocation_not_labeled_history():
    torch.manual_seed(3)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(6, 3)
    y = torch.randn(6, 3)

    def train_all(m):
        opt = torch.optim.SGD(m.parameters(), lr=0.2)
        losses = []
        for _ in range(4):
            opt.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return {"losses": losses}

    def train_none(m):
        return {"losses": [float(F.mse_loss(m(x), y).detach())] * 4}

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=x[:2],
        train_a=train_all,
        train_b=train_none,
        history_label_a="all",
        history_label_b="frozen",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={"sgd": 0.2},
        optimizer_config_b={"sgd": 0.2},
        seed_a=3,
        seed_b=3,
        subsequent_protocol_a={"trainable": "all"},
        subsequent_protocol_b={"trainable": "none"},
    )
    assert r.status == Status.UPDATE_ALLOCATION_EFFECT.value
    assert "HISTORY_EFFECT_PRESENT" in r.rejected_or_weakened


def test_strict_without_fingerprint_is_invalid():
    torch.manual_seed(4)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(4, 3)

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=x[:2],
        train_a=lambda m: {"losses": [0.0]},
        train_b=lambda m: {"losses": [0.0]},
        history_label_a="a",
        history_label_b="b",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={},
        optimizer_config_b={},
        seed_a=4,
        seed_b=4,
        strict=True,
    )
    assert r.status == Status.ASSAY_INVALID.value


def test_verified_same_protocol_momentum_is_history_present():
    torch.manual_seed(5)
    w = Tiny()
    x = torch.randn(6, 3)
    y = torch.randn(6, 3)
    optw = torch.optim.SGD(w.parameters(), lr=0.1, momentum=0.9)
    for _ in range(3):
        optw.zero_grad()
        F.mse_loss(w(x), y).backward()
        optw.step()
    a = Tiny()
    b = Tiny()
    a.load_state_dict(copy.deepcopy(w.state_dict()))
    b.load_state_dict(copy.deepcopy(w.state_dict()))
    oa = torch.optim.SGD(a.parameters(), lr=0.1, momentum=0.9)
    ob = torch.optim.SGD(b.parameters(), lr=0.1, momentum=0.9)
    oa.load_state_dict(copy.deepcopy(optw.state_dict()))

    def train_with(opt):
        def train(m):
            losses = []
            for _ in range(4):
                opt.zero_grad()
                loss = F.mse_loss(m(x), y)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            return {"losses": losses}

        return train

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=x[:2],
        train_a=train_with(oa),
        train_b=train_with(ob),
        history_label_a="MOM_WARM",
        history_label_b="MOM_FRESH",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={"sgd": 0.1, "momentum": 0.9},
        optimizer_config_b={"sgd": 0.1, "momentum": 0.9},
        seed_a=5,
        seed_b=5,
        optimizer_a=oa,
        optimizer_b=ob,
        strict=True,
    )
    assert r.extras["protocol_kind"] == "VERIFIED_PROTOCOL"
    assert r.status == Status.HISTORY_EFFECT_PRESENT.value


def test_verified_lr_mismatch_is_allocation_not_history():
    torch.manual_seed(6)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(6, 3)
    y = torch.randn(6, 3)
    oa = torch.optim.SGD(a.parameters(), lr=0.5)
    ob = torch.optim.SGD(b.parameters(), lr=0.01)

    def train_with(opt):
        def train(m):
            losses = []
            for _ in range(4):
                opt.zero_grad()
                loss = F.mse_loss(m(x), y)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            return {"losses": losses}

        return train

    r = history_shift(
        model_a=a,
        model_b=b,
        calibration_inputs=x[:2],
        train_a=train_with(oa),
        train_b=train_with(ob),
        history_label_a="fast",
        history_label_b="slow",
        task_spec_a={"t": 1},
        task_spec_b={"t": 1},
        optimizer_config_a={"lr": "declared_same"},
        optimizer_config_b={"lr": "declared_same"},
        seed_a=6,
        seed_b=6,
        optimizer_a=oa,
        optimizer_b=ob,
        strict=True,
    )
    assert r.status == Status.UPDATE_ALLOCATION_EFFECT.value
    assert r.extras["protocol_kind"] == "VERIFIED_PROTOCOL_MISMATCH"
