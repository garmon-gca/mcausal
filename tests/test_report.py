from __future__ import annotations

import json

from mcausal.report import to_html, to_json_dict, write_html, write_json
from mcausal.schemas import ProbeReport, Status


def test_json_roundtrip(tmp_path):
    r = ProbeReport(
        observed_effect="e",
        reproduced=True,
        current_function_match="match",
        candidate_cause="c",
        intervention="i",
        intervention_valid=True,
        residual="0",
        supported=["a"],
        rejected_or_weakened=["b"],
        remaining_alternatives=["d"],
        scope={"k": 1},
        provenance={"p": "q"},
        status=Status.EFFECT_ONLY.value,
    )
    p = write_json(r, tmp_path / "r.json")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    d = to_json_dict(r)
    assert loaded["observed_effect"] == d["observed_effect"]
    assert loaded["status"] == "EFFECT_ONLY"
    assert "supported" in loaded


def test_html_has_sections(tmp_path):
    r = ProbeReport(
        observed_effect="e",
        reproduced=True,
        current_function_match="match",
        candidate_cause="c",
        intervention="i",
        intervention_valid=False,
        residual="n/a",
        status=Status.CONSTRUCTION_NULL.value,
    )
    h = to_html(r, title="t")
    for sec in ("Observed", "Evidence", "Intervention", "Residual", "Conclusion", "Scope"):
        assert f"<h2>{sec}</h2>" in h
    p = write_html(r, tmp_path / "r.html")
    assert p.exists()
