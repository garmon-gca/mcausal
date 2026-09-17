from __future__ import annotations

import json

from mcausal.cli import main


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.1"


def test_version_subcommand(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.1"


def test_report_json_to_html(tmp_path):
    src = tmp_path / "p.json"
    src.write_text(
        json.dumps(
            {
                "observed_effect": "e",
                "reproduced": True,
                "current_function_match": "m",
                "candidate_cause": "c",
                "intervention": "i",
                "intervention_valid": True,
                "residual": "r",
                "supported": [],
                "rejected_or_weakened": [],
                "remaining_alternatives": [],
                "scope": {},
                "provenance": {},
                "status": "EFFECT_ONLY",
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "p.html"
    assert main(["report", str(src), "-o", str(dest)]) == 0
    html = dest.read_text(encoding="utf-8")
    assert "<h2>Observed</h2>" in html
    assert "EFFECT_ONLY" in html
