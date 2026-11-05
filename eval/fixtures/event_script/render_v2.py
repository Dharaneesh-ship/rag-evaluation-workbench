#!/usr/bin/env python3
"""rag-side renderer for the v2 sales data (附录 FR.4 point 3, 附录 FS/FU).

Imports MAEDA's ``scripts/generate_demo_data.py`` read-only and swaps its
event loader, so the render code is MAEDA's own, unforked. Writes only into
``--out-dir`` (refused when it lies inside the MAEDA or cleaner repository),
never into MAEDA's ``data/demo``.

Modes
-----
``v1-only``   the 12 batch-v1 events, in file order. Must reproduce MAEDA's
              current ``sales_data.csv`` byte for byte (acceptance check).
``v2``        v1 events + v2 dev (public) + v2 test (sealed).
``public``    v1 events + v2 dev only (what a machine without the sealed file
              can render; NOT a deliverable).
``baseline``  no events at all: the FU.2(2) no-event baseline. The renderer
              draws its random numbers before applying event multipliers, so
              this is row-for-row aligned with every other mode.

Output: ``sales_data.csv`` and ``render_receipt.json`` (hashes and counts only;
no event content) in ``--out-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.eval.split_guard import load_event_script_source, load_sealed_source  # noqa: E402

DEFAULT_MAEDA_SCRIPTS = Path.home() / "MAEDA" / "scripts"
FORBIDDEN_OUT_ROOTS = (Path.home() / "MAEDA", Path.home() / "agentic-data-cleaner-v2")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble_events(mode: str, public_events: list[dict], sealed_events: list[dict]) -> list[dict]:
    """Event list handed to the renderer for ``mode`` (order matters for rounding)."""
    v1 = [e for e in public_events if e.get("batch", "v1") == "v1"]
    v2_pub = [e for e in public_events if e.get("batch") == "v2"]
    if mode == "baseline":
        return []
    if mode == "v1-only":
        return v1
    if mode == "public":
        return v1 + v2_pub
    if mode == "v2":
        if not sealed_events:
            raise SystemExit("mode v2 needs the sealed file; it is absent on this machine")
        return v1 + sorted(v2_pub + sealed_events, key=lambda e: e["event_id"])
    raise SystemExit(f"unknown mode {mode!r}")


def refuse_forbidden_out_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    for root in FORBIDDEN_OUT_ROOTS:
        root = root.resolve()
        if resolved == root or root in resolved.parents:
            raise SystemExit(f"refusing to write inside {root}: this tool never writes to another repository")


def _git_head(repo: Path) -> str | None:
    res = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else None


def render(mode: str, out_dir: Path, maeda_scripts: Path) -> dict:
    refuse_forbidden_out_dir(out_dir)
    _, public = load_event_script_source(reason="render_v2.py needs every event to render the sales data")
    _, sealed = load_sealed_source(reason="render_v2.py renders the sealed test events into the data")
    events = assemble_events(mode, public.get("events", []), (sealed or {}).get("events", []))

    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(out_dir)  # generate_demo_data writes to the relative path data/demo
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(maeda_scripts))
    try:
        gen = importlib.import_module("generate_demo_data")
        gen.load_frozen_events = lambda: events
        gen.generate_sales()
    finally:
        os.chdir(cwd)
        sys.path.remove(str(maeda_scripts))

    produced = out_dir / "data" / "demo" / "sales_data.csv"
    target = out_dir / "sales_data.csv"
    produced.replace(target)
    # the module also created empty directories only; remove them
    for d in (out_dir / "data" / "demo", out_dir / "data"):
        try:
            d.rmdir()
        except OSError:
            pass

    renderer = maeda_scripts / "generate_demo_data.py"
    receipt = {
        "mode": mode,
        "events_rendered": len(events),
        "csv_sha256": sha256_file(target),
        "csv_rows": sum(1 for _ in target.open("rb")) - 1,
        "renderer_sha256": sha256_file(renderer),
        "renderer_maeda_head": _git_head(maeda_scripts.parent),
    }
    (out_dir / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["v1-only", "v2", "public", "baseline"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--maeda-scripts", type=Path, default=DEFAULT_MAEDA_SCRIPTS)
    args = ap.parse_args()
    print(json.dumps(render(args.mode, args.out_dir, args.maeda_scripts), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
