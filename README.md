# mcausal

**mcausal is a PyTorch causal training debugger.**

It helps determine not only whether training behavior changed, but whether a suspected training-state variable causally explains that change.

Version **0.1.0**. License: Apache-2.0. Author: **garmon-gca**.

Verified on **Python 3.11**. Other Python versions are not claimed.

## What it does

**UpdateReality** — one training step as a chain:

`gradient → optimizer update → Δparameters → Δfunction`

A small `|gradient|` is not automatically small learning.

**HistoryShift** — after matching the *current* function, do two systems learn the *next* task differently because of different training history? Same subsequent protocol required. Different trainable sets are **not** history; they are update allocation.

**ResidualMatch** — you name a suspected mediator and an intervention:

`observed effect → suspected mediator → match/intervene → residual`

Outcomes include support, rejection, residual remaining, construction-null, inconclusive. There is no causal score 0–100.

## What mcausal is not

mcausal is **not**:

- an automatic causal oracle
- a universal root-cause detector
- an intelligence benchmark
- a general mechanistic-interpretability framework
- a replacement for loss / eval / profiling
- a guarantee of causal inference
- a proven tool for all architectures, optimizers, or tasks

## Quickstart

```text
git clone https://github.com/garmon-gca/mcausal.git
cd mcausal
python -m venv .venv
.venv\Scripts\activate
pip install ".[dev]"
mcausal --version
mcausal ref
python -m pytest
```

On Unix: `source .venv/bin/activate`.

Minimal `UpdateReality` example: `examples/minimal_pytorch/run_update_reality.py`.

## API

```python
from mcausal import update_reality, history_shift, residual_match, write_json, write_html

report = update_reality(model, optimizer, batch, calibration_inputs, loss_fn=loss_fn)
```

CLI:

```text
mcausal --version
mcausal ref [outdir]
mcausal report probe.json -o probe.html
```

`mcausal ref` **runs** R1/R2/R3. It does not print a stored PASS.

## Evidence (short)

Internal tests, not a public benchmark:

| Case | Result |
|---|---|
| R1 causal positive | PASS |
| R2 construction-null | PASS |
| R3 Adam metric trap | PASS |
| GPT-2 + LoRA optimizer-state | residual collapsed to 0 after copying Adam state |
| DistilBERT + LoRA (independent stack) | same class reproduced |
| Trainable-set / update allocation | `UPDATE_ALLOCATION_EFFECT` / `TRAINABLE_SET_MEDIATION` (not HistoryShift) |
| Replication pack R4–R8 | R4/R5/R8 PASS; **R6 and R7 PARTIAL** (some seeds `INCONCLUSIVE` under prereg thresholds) |

PARTIAL is not PASS. Details: `docs/EVIDENCE.md`, `docs/CLAIMS_AND_LIMITATIONS.md`.

External usability: **NOT YET**. See `docs/EXTERNAL_USABILITY_GATE.md`.

## Feedback

Feedback and reproducible bug reports are welcome.

Substantial external code contributions are **not** automatically accepted. The immediate need is users and reports, not a contribution pipeline.

Do not attach proprietary datasets to issues.

## Citation

See `CITATION.cff`.
