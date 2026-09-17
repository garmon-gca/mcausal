"""Machine-readable probe reports. No causal score 0–100."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class Status(str, Enum):
    EFFECT_ONLY = "EFFECT_ONLY"
    HISTORY_EFFECT_PRESENT = "HISTORY_EFFECT_PRESENT"
    CAUSAL_VARIABLE_SUPPORTED = "CAUSAL_VARIABLE_SUPPORTED"
    CAUSAL_VARIABLE_REJECTED = "CAUSAL_VARIABLE_REJECTED"
    RESIDUAL_REMAINS = "RESIDUAL_REMAINS"
    CONSTRUCTION_NULL = "CONSTRUCTION_NULL"
    ASSAY_INVALID = "ASSAY_INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_DETECTED_HISTORY_EFFECT = "NO_DETECTED_HISTORY_EFFECT"
    CURRENT_FUNCTION_NOT_MATCHED = "CURRENT_FUNCTION_NOT_MATCHED"
    UPDATE_ALLOCATION_EFFECT = "UPDATE_ALLOCATION_EFFECT"
    TRAINABLE_SET_MEDIATION = "TRAINABLE_SET_MEDIATION"


ALLOWED_STATUS = {s.value for s in Status}


def jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


@dataclass
class ProbeReport:
    observed_effect: str
    reproduced: bool
    current_function_match: str
    candidate_cause: str
    intervention: str
    intervention_valid: bool
    residual: str
    supported: list[str] = field(default_factory=list)
    rejected_or_weakened: list[str] = field(default_factory=list)
    remaining_alternatives: list[str] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    status: str = Status.INCONCLUSIVE.value
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUS:
            raise ValueError(f"status {self.status!r} not in {sorted(ALLOWED_STATUS)}")

    def to_dict(self) -> dict[str, Any]:
        d = jsonable(self)
        # extras folded for the published schema; keep extras too
        return d


def make_report(**kwargs: Any) -> ProbeReport:
    return ProbeReport(**kwargs)
