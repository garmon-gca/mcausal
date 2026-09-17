# How it works

Three cores. No fourth probe in 0.1.2.

## UpdateReality

One `loss.backward()` + `optimizer.step()`, instrumented.

**Ground truth:** actual Δparameter (and Δfunction on the supplied calibration inputs).

**Estimate:** `estimated_update_norm` is computed only for an explicit subset (plain SGD; Adam with `weight_decay=0`; AdamW). Clipping, if requested, is applied **before** the estimate. Coupled Adam weight decay, SGD Nesterov/dampening, AMSGrad, and other classes are `UNSUPPORTED_FOR_EXACT_UPDATE_ESTIMATE` — not a silent approximation.

## HistoryShift

Function comparison is an **empirical match on the supplied calibration inputs**, not global equality of functions. The report stores calibration shape / n.

`optimizer_config` and `subsequent_protocol` are **DECLARED**. They are not proof `train_a`/`train_b` are identical.

- No measured fingerprint → trajectory gap is `HISTORY_EFFECT_CANDIDATE` (`PROTOCOL_UNVERIFIED`), never unqualified `HISTORY_EFFECT_PRESENT`.
- Fingerprint match (`optimizer_a/b` or `protocol_fingerprint_a/b`) → `VERIFIED_PROTOCOL`; then a gap may be `HISTORY_EFFECT_PRESENT`.
- Fingerprint mismatch → `UPDATE_ALLOCATION_EFFECT` (not history).
- `strict=True` without a fingerprint → `ASSAY_INVALID`.

## ResidualMatch

**mcausal tests a user-specified causal hypothesis; it does not discover the true cause automatically.**

You supply measurements and the intervention. The library reports whether matching that hypothesis collapsed the effect.
