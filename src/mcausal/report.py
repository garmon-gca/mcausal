"""JSON + one-page static HTML. Not a web app."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .schemas import ProbeReport, jsonable


SCHEMA_KEYS = (
    "observed_effect",
    "reproduced",
    "current_function_match",
    "candidate_cause",
    "intervention",
    "intervention_valid",
    "residual",
    "supported",
    "rejected_or_weakened",
    "remaining_alternatives",
    "scope",
    "provenance",
)


def to_json_dict(report: ProbeReport | dict[str, Any]) -> dict[str, Any]:
    d = report.to_dict() if isinstance(report, ProbeReport) else jsonable(report)
    out = {k: d.get(k) for k in SCHEMA_KEYS}
    out["status"] = d.get("status")
    if "extras" in d:
        out["extras"] = d["extras"]
    return out


def write_json(report: ProbeReport | dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _pre(obj: Any) -> str:
    txt = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    return html.escape(txt)


def to_html(report: ProbeReport | dict[str, Any], title: str = "mcausal report") -> str:
    d = to_json_dict(report)
    status = html.escape(str(d.get("status", "")))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Georgia, serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; color: #111; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.05rem; margin-top: 1.6rem; border-bottom: 1px solid #ccc; }}
.badge {{ display: inline-block; padding: .15rem .5rem; border: 1px solid #333; font-family: Consolas, monospace; }}
pre {{ background: #f6f6f4; padding: .8rem; overflow: auto; font-size: .85rem; }}
.k {{ color: #444; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="badge">{status}</p>
<h2>Observed</h2>
<pre>{_pre(d.get("observed_effect"))}</pre>
<p class="k">reproduced: {html.escape(str(d.get("reproduced")))}</p>
<p class="k">current_function_match: {html.escape(str(d.get("current_function_match")))}</p>
<h2>Evidence</h2>
<pre>{_pre({"candidate_cause": d.get("candidate_cause"), "supported": d.get("supported"), "rejected_or_weakened": d.get("rejected_or_weakened"), "remaining_alternatives": d.get("remaining_alternatives")})}</pre>
<h2>Intervention</h2>
<pre>{_pre({"intervention": d.get("intervention"), "intervention_valid": d.get("intervention_valid")})}</pre>
<h2>Residual</h2>
<pre>{_pre(d.get("residual"))}</pre>
<h2>Conclusion</h2>
<pre>{_pre(d.get("status"))}</pre>
<h2>Scope</h2>
<pre>{_pre({"scope": d.get("scope"), "provenance": d.get("provenance")})}</pre>
</body>
</html>
"""


def write_html(report: ProbeReport | dict[str, Any], path: str | Path, title: str = "mcausal report") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_html(report, title=title), encoding="utf-8")
    return path
