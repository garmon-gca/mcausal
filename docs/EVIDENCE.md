# Evidence (sanitized)

Internal studies. Original local corpora and checkpoints are **not** published. These summaries are not a claim of full external reproducibility of the LoRA stacks.

## Reference suite (reproducible in this repo)

| ID | Meaning | Result |
|---|---|---|
| R1 | Causal positive (gain mediates new-task speed on a tiny supervised path) | PASS |
| R2 | Construction-null (different labels, identical trajectory) | PASS |
| R3 | Adam metric trap (`|g|` vs Δθ at real ε) | PASS |

Run: `mcausal ref`. Computation is live.

## GPT-2 + LoRA (internal)

Causal LM, LoRA on `c_attn`. Two systems: identical LoRA weights and t0 eval/logits (exact function match); identical first-step gradient; different subsequent losses. Copying Adam state collapsed residual to **0**. Ordinary t0 eval/grad did not show the gap.

## DistilBERT + LoRA (internal, independent stack)

Encoder classifier, LoRA on `q_lin`/`v_lin`, different data/task. Same optimizer-state class reproduced. Copying Adam state collapsed residual.

## Different causal class (not HistoryShift)

HEAD trainable vs LoRA-only changes **which parameters may update** on the new task. Label: `UPDATE_ALLOCATION_EFFECT`. ResidualMatch after matching the trainable set: `TRAINABLE_SET_MEDIATION`. This is **not** a history-only contrast.

## Replication pack (internal)

| Stand | Architecture | Class | Status |
|---|---|---|---|
| R4 | MLP | stateless SGD negative | **PASS** (16/16 `CONSTRUCTION_NULL`, gap 0) |
| R5 | MLP | SGD momentum state | **PASS** (copy momentum, residual 0) |
| R6 | GRU | Adam state | **PARTIAL** |
| R7 | CNN | update allocation | **PARTIAL** |
| R8 | MLP continual | replay-buffer state | **PASS** (copy replay, residual 0) |

**PARTIAL** means some seeds did not reach the preregistered mediation threshold and were classified `INCONCLUSIVE`. It is not PASS.

## What this does not show

- All architectures
- All optimizers
- Public LoRA datasets
- External-user success
