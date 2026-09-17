# USER_RECIPE — two runs behave differently. What now?

You have this package and these docs. Frozen API:

`update_reality`, `history_shift`, `residual_match`, `write_json`, `write_html`

mcausal will **not** name the mediator for you.

## Sequence

1. **Current-function equivalence** — same inputs, compare outputs/logits. If they differ, you are not looking at “history with matched function.” HistoryShift will return `CURRENT_FUNCTION_NOT_MATCHED`.
2. **Observe subsequent effect** — train both on the **same** next-task protocol. Log loss/eval. If the *protocol* differs (other trainable set, other clip), that is allocation, not history.
3. **Inspect update reality** — wrap one real step. Read whether `|g|`, proposed update, Δθ, and Δfunction agree.
4. **State a suspected mediator** — optimizer buffers, trainable set, replay/data state, clip, … This step is **your hypothesis**. It is not always knowable.
5. **Intervene / match** — copy or equalize that state only. Keep weights/function invariants.
6. **Measure residual** — same next-task protocol after the match.
7. **Report** — supported / rejected / residual remains / inconclusive. Then decide what to change in training (reset optimizer state, freeze a module, stop treating `|g|` as step size, …).

## HistoryShift (same subsequent protocol)

```python
from mcausal import history_shift

proto = {"trainable": "lora+head", "clip": None, "lr": 2e-4}
hs = history_shift(
    model_a=model_a, model_b=model_b,
    calibration_inputs=calib,
    train_a=train_fn, train_b=train_fn,
    history_label_a="history_A", history_label_b="history_B",
    task_spec_a={"task": "your_task"}, task_spec_b={"task": "your_task"},
    optimizer_config_a={"name": "Adam", "lr": 2e-4},
    optimizer_config_b={"name": "Adam", "lr": 2e-4},
    seed_a=0, seed_b=0,
    subsequent_protocol_a=proto, subsequent_protocol_b=proto,
)
```

## ResidualMatch

```python
from mcausal import residual_match, write_json, write_html

rm = residual_match(
    suspected_variable="your_guess",
    measure_effect=lambda: effect_before,
    measure_mediator=lambda: {"note": "what you measured"},
    intervene=lambda: {"op": "what you actually matched"},
    measure_invariants=lambda: {"function_matched": True},
    invariant_ok=lambda inv: inv["function_matched"],
    measure_effect_after=lambda: effect_after,
)
write_json(rm, "mcausal_residual.json")
write_html(rm, "mcausal_residual.html", title="ResidualMatch")
```

If you cannot hook these functions without editing mcausal internals, that is a **usability failure**. Do not invent a new probe.
