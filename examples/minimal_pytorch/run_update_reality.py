#!/usr/bin/env python3
"""Minimal UpdateReality example. No Hugging Face. CPU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcausal import update_reality, write_html, write_json


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)

    def forward(self, x):
        return self.lin(x)


def main() -> int:
    torch.manual_seed(0)
    model = Tiny()
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    calib = x[:4]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

    def loss_fn(m, batch):
        xb, yb = batch
        return F.cross_entropy(m(xb), yb)

    report = update_reality(model, opt, (x, y), calib, loss_fn=loss_fn)
    write_json(report, "update_reality_example.json")
    write_html(report, "update_reality_example.html", title="UpdateReality example")
    print("levels_diverge", report["levels_diverge"])
    print("gradient_norm", round(report["gradient_norm"], 4))
    print("actual_parameter_movement", round(report["actual_parameter_movement"], 4))
    print("wrote update_reality_example.json and .html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
