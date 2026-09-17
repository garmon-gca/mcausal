# How it works

Three cores. No fourth probe in 0.1.0.

## UpdateReality

One `loss.backward()` + `optimizer.step()`, instrumented.

Reports gradient norm, optimizer-proposed update, actual Δθ, function movement on calibration inputs, and ratios between those levels. Flags when `|g|` is a bad size meter (Adam ε, clip, weight decay).

## HistoryShift

Requires a matched **current** function (or it returns `CURRENT_FUNCTION_NOT_MATCHED`).

Valid as **history** only if the *subsequent* protocol is the same (trainable set, clip, lr groups, new-task data). Declare `subsequent_protocol_a/b`. If those dicts differ, a gap is `UPDATE_ALLOCATION_EFFECT`, not `HISTORY_EFFECT_PRESENT`.

Identical labels-but-wait: different history *labels* with identical trajectories → `CONSTRUCTION_NULL`.

## ResidualMatch

You supply measurements and the intervention. The library does not invent the mediator. It reports whether matching it collapsed the effect, left a residual, failed invariants, or was inconclusive.
