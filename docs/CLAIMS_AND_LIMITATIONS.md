# Claims and limitations

Rule: every public claim has a concrete evidence source. If a sentence needs “probably / likely / potentially” without that source, delete it.

## Allowed

mcausal 0.1.1 (same cores as 0.1.0):

- is a Python / PyTorch causal-training diagnostic library (`pyproject.toml`, this repo)
- contains three frozen cores: UpdateReality, HistoryShift, ResidualMatch (`src/mcausal/`)
- runs reference cases R1/R2/R3 at call time (`mcausal ref`; `src/mcausal/reference/`)
- can return positive, null, and inconclusive outcomes (`src/mcausal/schemas.py`)
- was exercised on several **internal** experimental stacks (see `docs/EVIDENCE.md`)
- identified optimizer-state mediation on internal GPT-2 LoRA and DistilBERT LoRA runs
- identified update-allocation / trainable-set mediation on an internal DistilBERT and CNN stand
- identified replay / training-data-state mediation on an internal continual stand
- in weaker seeds, did **not** confirm mediation (`INCONCLUSIVE`; R6/R7 PARTIAL)

## Forbidden (do not write these)

- “mcausal automatically finds the true cause”
- “works universally on neural networks”
- “works on any architecture”
- “prevents forgetting”
- “makes training safe”
- “improves training” as a general law
- “better than Cockpit / Captum / TorchLens / MIB” without a direct benchmark
- “production ready”
- “industry proven”
- “scientifically proven universal causal debugger”

## External usability

**NOT YET.** A stranger using their own pipeline, with only this package and docs, has not closed that gate.
