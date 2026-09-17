"""python -m mcausal.reference_suite"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mcausal.reference import r1_gain_causal, r2_construction_null, r3_adam_metric_trap
from mcausal.report import write_html, write_json


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out_dir = Path("reports")
    if argv:
        out_dir = Path(argv[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("R1 CAUSAL POSITIVE", r1_gain_causal),
        ("R2 CONSTRUCTION NULL", r2_construction_null),
        ("R3 METRIC TRAP", r3_adam_metric_trap),
    ]
    all_ok = True
    summary = []
    for label, mod in cases:
        report = mod.run()
        ok = mod.passes(report)
        tag = "PASS" if ok else "FAIL"
        print(f"{label}: {tag}")
        if not ok:
            all_ok = False
        slug = label.split()[0].lower()
        write_json(report, out_dir / f"{slug}.json")
        write_html(report, out_dir / f"{slug}.html", title=label)
        summary.append({"label": label, "result": tag, "status": report.status})
    (out_dir / "reference_suite.json").write_text(
        json.dumps({"all_pass": all_ok, "cases": summary}, indent=2), encoding="utf-8"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
