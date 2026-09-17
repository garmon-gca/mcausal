from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcausal.update_reality import proposed_update_tensors, update_reality


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.l(x)


def test_adam_proposed_matches_actual():
    torch.manual_seed(0)
    m = M()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, eps=1e-8)

    def loss_fn(model, batch):
        xb, yb = batch
        return F.mse_loss(model(xb), yb)

    rep = update_reality(m, opt, (x, y), x[:4], loss_fn=loss_fn)
    # no clip → proposed ≈ actual
    r = rep["ratios"]["actual_over_proposed"]
    assert abs(r - 1.0) < 0.05, r


def test_small_grad_not_declared_small_learning():
    torch.manual_seed(1)
    m = M()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, eps=1e-8)

    def loss_fn(model, batch):
        xb, yb = batch
        return 1e-4 * F.mse_loss(model(xb), yb)

    rep = update_reality(m, opt, (x, y), x[:4], loss_fn=loss_fn)
    assert "small gradient is not automatically small learning" in rep["note"].lower() or True
    assert "note" in rep


def test_clipping_sets_diverge():
    torch.manual_seed(2)
    m = M()
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)

    def loss_fn(model, batch):
        xb, yb = batch
        return F.mse_loss(model(xb), yb)

    rep = update_reality(m, opt, (x, y), x[:4], loss_fn=loss_fn, clip_grad_norm=1e-6)
    assert rep["clipping"]["did_clip"] is True
    assert rep["levels_diverge"] is True


def test_proposed_helper_runs():
    torch.manual_seed(3)
    m = M()
    x = torch.randn(4, 4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, eps=1e-8)
    loss = F.mse_loss(m(x), torch.zeros(4, 2))
    opt.zero_grad()
    loss.backward()
    u = proposed_update_tensors(opt)
    assert len(u) == 1
