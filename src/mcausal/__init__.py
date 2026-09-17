"""mcausal v0.1 — causal training debugger (PyTorch). Frozen API."""

from .history_shift import history_shift
from .report import to_html, to_json_dict, write_html, write_json
from .residual_match import residual_match
from .schemas import ProbeReport, Status
from .update_reality import update_reality

__version__ = "0.1.1"
__all__ = [
    "update_reality",
    "history_shift",
    "residual_match",
    "ProbeReport",
    "Status",
    "write_json",
    "write_html",
    "to_json_dict",
    "to_html",
    "__version__",
]
