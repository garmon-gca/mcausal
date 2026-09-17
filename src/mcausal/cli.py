"""Minimal CLI for the frozen v0.1 package. Not a platform."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mcausal import __version__


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_ref(args: argparse.Namespace) -> int:
    from mcausal.reference_suite import main as ref_main

    out = [args.out] if args.out else []
    return ref_main(out)


def _cmd_report(args: argparse.Namespace) -> int:
    from mcausal.report import to_html, write_html

    src = Path(args.json)
    data = json.loads(src.read_text(encoding="utf-8"))
    dest = Path(args.out) if args.out else src.with_suffix(".html")
    write_html(data, dest, title=args.title or src.stem)
    if args.stdout:
        sys.stdout.write(to_html(data, title=args.title or src.stem))
    else:
        print(dest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcausal",
        description="mcausal v0.1 — causal training debugger (PyTorch)",
    )
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="cmd")

    s_ver = sub.add_parser("version", help="print package version")
    s_ver.set_defaults(fn=_cmd_version)

    s_ref = sub.add_parser("ref", help="run R1/R2/R3 reference suite")
    s_ref.add_argument("out", nargs="?", default="reports", help="output directory")
    s_ref.set_defaults(fn=_cmd_ref)

    s_rep = sub.add_parser("report", help="JSON probe → static HTML")
    s_rep.add_argument("json", help="probe JSON path")
    s_rep.add_argument("-o", "--out", default=None, help="HTML path")
    s_rep.add_argument("--title", default=None)
    s_rep.add_argument("--stdout", action="store_true")
    s_rep.set_defaults(fn=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False) and not args.cmd:
        return _cmd_version(args)
    fn = getattr(args, "fn", None)
    if fn is None:
        parser.print_help()
        return 0
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
