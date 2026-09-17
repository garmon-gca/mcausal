# Changelog

## 0.1.2

Hardening patch. No new probes.

- HistoryShift: declared protocol is not proof; `HISTORY_EFFECT_PRESENT` requires a verified fingerprint; otherwise `HISTORY_EFFECT_CANDIDATE` / `PROTOCOL_UNVERIFIED`; `strict=True` available
- UpdateReality: `actual_parameter_movement` is ground truth; optimizer prediction is `estimated_update` and only exact for a supported subset; clip applied before the estimate; coupled Adam WD / SGD Nesterov / dampening / AMSGrad → `UNSUPPORTED_FOR_EXACT_UPDATE_ESTIMATE`
- Function match wording: empirical match on supplied calibration inputs
- ResidualMatch code unchanged; docs state it tests a user hypothesis, not an automatic cause

## 0.1.1

Public-ready patch. Metadata and attribution only.

- public author unified to `Mangust`
- package version `0.1.1`
- no causal-core changes
- no scientific-behavior changes
- no new probes

`v0.1.0` remains the historical freeze.

## 0.1.0

First freeze.

- Three cores: UpdateReality, HistoryShift, ResidualMatch
- CLI: `mcausal --version`, `mcausal ref`, `mcausal report`
- Reference suite R1 / R2 / R3 executed at runtime
- Status vocabulary includes null, inconclusive, allocation (not only positive history effects)
