"""UpdateReality: raw gradient → optimizer update → Δparameter → Δfunction."""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .schemas import ProbeReport, Status, jsonable


def _outputs_to_tensor(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    if isinstance(out, dict):
        for k in ("logits", "last_hidden_state", "loss"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"cannot extract tensor from model output type {type(out)}")


def _finite_norm(t: torch.Tensor) -> float:
    n = float(t.detach().float().norm().cpu())
    return n if math.isfinite(n) else float("nan")


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def param_delta_norm(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    acc = 0.0
    for k, b in before.items():
        if k not in after:
            continue
        d = after[k].float() - b.float()
        acc += float(d.pow(2).sum())
    return math.sqrt(acc)


def function_delta(model: nn.Module, calibration_inputs: Any, y0: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        y1 = _outputs_to_tensor(call_model(model, calibration_inputs)).detach().float()
    y0f = y0.detach().float()
    diff = y1 - y0f
    rms = float(diff.pow(2).mean().sqrt())
    mx = float(diff.abs().max())
    a = y0f.reshape(-1)
    b = y1.reshape(-1)
    if float(a.norm()) < 1e-12 or float(b.norm()) < 1e-12:
        cos = float("nan")
    else:
        cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return {"rms": rms, "max_abs": mx, "cosine": cos, "l2": float(diff.norm())}


def call_model(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return model(**inputs)
    if hasattr(inputs, "keys") and "input_ids" in getattr(inputs, "keys")():
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _as_args(calibration_inputs: Any) -> tuple:
    if isinstance(calibration_inputs, tuple):
        return calibration_inputs
    return (calibration_inputs,)


def _grad_norms(model: nn.Module) -> tuple[float, dict[str, float]]:
    total = 0.0
    per: dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            per[n] = 0.0
            continue
        g = float(p.grad.detach().float().norm().cpu())
        per[n] = g
        total += g * g
    return math.sqrt(total), per


@torch.no_grad()
def proposed_update_tensors(optimizer: torch.optim.Optimizer) -> dict[int, torch.Tensor]:
    """What Adam/SGD/AdamW will apply on the next step(), given current grads + state."""
    out: dict[int, torch.Tensor] = {}
    opt_name = type(optimizer).__name__
    decoupled = isinstance(optimizer, torch.optim.AdamW) or "AdamW" in opt_name
    is_adam = isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)) or "Adam" in opt_name
    is_sgd = isinstance(optimizer, torch.optim.SGD) or opt_name == "SGD"

    for group in optimizer.param_groups:
        lr = float(group.get("lr", 0.0))
        wd = float(group.get("weight_decay", 0.0))
        maximize = bool(group.get("maximize", False))
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if maximize:
                grad = -grad
            if is_sgd:
                upd = grad
                if wd != 0.0:
                    upd = upd + wd * p.detach()
                momentum = float(group.get("momentum", 0.0))
                if momentum != 0.0:
                    buf = optimizer.state[p].get("momentum_buffer")
                    if buf is None:
                        buf = grad.clone()
                    else:
                        buf = buf.detach() * momentum + grad
                    upd = buf
                out[id(p)] = -lr * upd
                continue
            if is_adam:
                beta1, beta2 = group.get("betas", (0.9, 0.999))
                eps = float(group.get("eps", 1e-8))
                state = optimizer.state[p]
                step = int(state.get("step", 0)) + 1
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(p)
                    exp_avg_sq = torch.zeros_like(p)
                else:
                    exp_avg = exp_avg.detach()
                    exp_avg_sq = exp_avg_sq.detach()
                exp_avg = exp_avg * beta1 + (1.0 - beta1) * grad
                exp_avg_sq = exp_avg_sq * beta2 + (1.0 - beta2) * grad * grad
                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt() / math.sqrt(bias2) + eps
                step_size = lr / bias1
                upd = -step_size * exp_avg / denom
                if wd != 0.0:
                    if decoupled:
                        upd = upd - lr * wd * p.detach()
                    else:
                        # coupled: already in grad if caller added it; PyTorch Adam applies
                        # weight decay to grad before moments when foreach/fused off:
                        # we approximate decoupled-off as extra -lr*wd*p (common)
                        upd = upd - lr * wd * p.detach()
                out[id(p)] = upd
                continue
            out[id(p)] = -lr * grad
    return out


def _update_norm(updates: dict[int, torch.Tensor]) -> float:
    acc = 0.0
    for t in updates.values():
        acc += float(t.detach().float().pow(2).sum().cpu())
    return math.sqrt(acc)


def _optimizer_meta(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    g0 = optimizer.param_groups[0] if optimizer.param_groups else {}
    steps = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if "step" in st:
                try:
                    steps.append(int(st["step"]))
                except Exception:
                    pass
    return {
        "type": type(optimizer).__name__,
        "lr": g0.get("lr"),
        "eps": g0.get("eps"),
        "betas": g0.get("betas"),
        "weight_decay": g0.get("weight_decay"),
        "momentum": g0.get("momentum"),
        "amsgrad": g0.get("amsgrad"),
        "n_param_groups": len(optimizer.param_groups),
        "max_state_step": max(steps) if steps else 0,
    }


def update_reality(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    calibration_inputs: Any,
    loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
    clip_grad_norm: Optional[float] = None,
    train_mode: bool = True,
) -> dict[str, Any]:
    """One training step, instrumented.

    `loss_fn(model, batch) -> scalar`. If omitted, `batch` must be a precomputed
    scalar loss tensor (not recommended).
    """
    if train_mode:
        model.train()
    else:
        model.eval()

    warnings: list[str] = []
    before = snapshot_params(model)
    model.eval()
    with torch.no_grad():
        y0 = _outputs_to_tensor(call_model(model, calibration_inputs)).detach().clone()
    if train_mode:
        model.train()

    optimizer.zero_grad(set_to_none=True)
    if loss_fn is None:
        if not torch.is_tensor(batch):
            raise TypeError("batch must be a scalar loss if loss_fn is None")
        loss = batch
    else:
        loss = loss_fn(model, batch)
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar tensor")
    loss.backward()

    grad_norm_raw, per_raw = _grad_norms(model)
    proposed = proposed_update_tensors(optimizer)
    proposed_norm = _update_norm(proposed)

    clip_info: dict[str, Any] | None = None
    if clip_grad_norm is not None:
        unclipped = grad_norm_raw
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        grad_norm_clipped, per_clip = _grad_norms(model)
        clip_info = {
            "max_norm": clip_grad_norm,
            "unclipped_grad_norm": unclipped,
            "clipped_grad_norm": grad_norm_clipped,
            "did_clip": unclipped > clip_grad_norm + 1e-12,
        }
        grad_norm = grad_norm_clipped
        per = per_clip
    else:
        grad_norm = grad_norm_raw
        per = per_raw

    optimizer.step()
    after = snapshot_params(model)
    actual_norm = param_delta_norm(before, after)
    fdelta = function_delta(model, calibration_inputs, y0)
    meta = _optimizer_meta(optimizer)
    lr = float(meta.get("lr") or 0.0)
    grad_times_lr = grad_norm * lr

    def _ratio(a: float, b: float) -> float:
        if not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-15:
            return float("nan")
        return a / b

    ratios = {
        "actual_delta_over_grad": _ratio(actual_norm, grad_norm),
        "proposed_over_grad": _ratio(proposed_norm, grad_norm),
        "actual_over_proposed": _ratio(actual_norm, proposed_norm),
        "actual_over_grad_times_lr": _ratio(actual_norm, grad_times_lr),
        "function_l2_over_actual_delta": _ratio(fdelta["l2"], actual_norm),
        "function_l2_over_grad": _ratio(fdelta["l2"], grad_norm),
    }
    # Divergence of *levels*: |g| is not a learning-size meter if Adam/clip/wd intervene.
    level_diverge = False
    r = ratios["actual_over_grad_times_lr"]
    if math.isfinite(r) and (r < 0.2 or r > 5.0):
        level_diverge = True
        warnings.append("actual Δθ is not ~ lr·‖g‖; do not read small/large grad as small/large learning")
    r2 = ratios["actual_over_proposed"]
    if math.isfinite(r2) and abs(r2 - 1.0) > 0.15:
        level_diverge = True
        warnings.append("proposed optimizer update ≠ actual Δparameter (clip / hook / skipped params)")
    if clip_info and clip_info.get("did_clip"):
        level_diverge = True

    if not math.isfinite(float(loss.detach())):
        warnings.append("non-finite loss")

    report = {
        "gradient_norm": grad_norm,
        "gradient_norm_raw": grad_norm_raw,
        "gradient_norm_per_param": per,
        "optimizer_proposed_update_norm": proposed_norm,
        "actual_parameter_movement": actual_norm,
        "output_function_movement": fdelta,
        "ratios": ratios,
        "levels_diverge": level_diverge,
        "optimizer": meta,
        "clipping": clip_info,
        "weight_decay": meta.get("weight_decay"),
        "epsilon": meta.get("eps"),
        "loss": float(loss.detach()),
        "warnings": warnings,
        "note": "A small gradient is not automatically small learning.",
    }
    return jsonable(report)


def update_reality_probe(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    calibration_inputs: Any,
    loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
    **kwargs: Any,
) -> ProbeReport:
    raw = update_reality(model, optimizer, batch, calibration_inputs, loss_fn=loss_fn, **kwargs)
    diverge = bool(raw["levels_diverge"])
    return ProbeReport(
        observed_effect="level chain grad → proposed update → Δθ → Δfunction",
        reproduced=True,
        current_function_match="n/a (single system, one step)",
        candidate_cause="optimizer mechanics (Adam ε, clip, weight decay) vs raw |g|",
        intervention="instrumented optimizer.step",
        intervention_valid=True,
        residual="see ratios",
        supported=["levels_diverge"] if diverge else [],
        rejected_or_weakened=["|g| as learning-size meter"] if diverge else [],
        remaining_alternatives=["data noise", "architecture scale"],
        scope={"api": "update_reality"},
        provenance={"component": "UpdateReality"},
        status=Status.EFFECT_ONLY.value,
        extras=raw,
    )
