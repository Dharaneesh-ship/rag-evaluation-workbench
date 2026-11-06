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

_ES = Path(__file__).resolve().parent
if str(_ES) not in sys.path:
    sys.path.insert(0, str(_ES))

from rag.eval.split_guard import load_event_script_source, load_sealed_source  # noqa: E402

DEFAULT_MAEDA_REPO = Path.home() / "MAEDA"
DEFAULT_MAEDA_SCRIPTS = DEFAULT_MAEDA_REPO / "scripts"
#: 附录 FZ.4: the renderer the delivered CSV was produced with. MAEDA's current
#: HEAD no longer generates data (it receives this CSV by hash), so the renderer
#: is read out of MAEDA's git history with ``git show`` (read-only) and pinned.
PINNED_RENDERER_REF = "d557e52"
PINNED_RENDERER_SHA256 = "ed223821bc709718e13650c01bd59bf71cb06a974b3040b5cf2730334451de18"
RENDERER_PATH_IN_REPO = "scripts/generate_demo_data.py"
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


def _check_against_manifest(public: dict, sealed: dict) -> dict:
    """A v2 deliverable may only be rendered from content that matches the frozen manifest."""
    manifest = json.loads((Path(__file__).parent / "event_script.manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("git_tag"):
        raise SystemExit("manifest is not frozen; refusing to render a v2 deliverable")
    from v2_checks import canonical_sha256, combined_sha256

    pub, seal = canonical_sha256(public), canonical_sha256(sealed)
    if (pub, seal, combined_sha256(pub, seal)) != (
        manifest["public_sha256"], manifest["sealed_sha256"], manifest["combined_sha256"]
    ):
        raise SystemExit("event content differs from the frozen manifest; refusing to render")
    return {k: manifest[k] for k in ("git_tag", "frozen_at", "public_sha256", "sealed_sha256", "combined_sha256", "slot_plan_sha256")}


def materialize_pinned_renderer(maeda_repo: Path, ref: str, expected_sha256: str, into: Path) -> tuple[Path, str]:
    """Write ``ref:scripts/generate_demo_data.py`` into ``into`` (outside any repo), verify its hash.

    Returns ``(directory, full commit hash of ref)``. Nothing in the MAEDA repo is touched.
    """
    res = subprocess.run(
        ["git", "-C", str(maeda_repo), "show", f"{ref}:{RENDERER_PATH_IN_REPO}"], capture_output=True
    )
    if res.returncode != 0:
        raise SystemExit(f"cannot read {ref}:{RENDERER_PATH_IN_REPO}: {res.stderr.decode().strip()}")
    got = hashlib.sha256(res.stdout).hexdigest()
    if got != expected_sha256:
        raise SystemExit(f"renderer at {ref} has sha256 {got}, expected {expected_sha256}")
    into.mkdir(parents=True, exist_ok=True)
    (into / "generate_demo_data.py").write_bytes(res.stdout)
    commit = subprocess.run(
        ["git", "-C", str(maeda_repo), "rev-parse", f"{ref}^{{commit}}"], capture_output=True, text=True
    ).stdout.strip()
    return into, commit


def check_clean_consistency(clean_csv: Path, dirty_csv: Path) -> dict:
    """Counts only. Relates the clean table to the quality-noise table (附录 GP.4 (2)).

    Expected: the first ``len(clean)`` dirty rows equal the clean rows except for
    revenue cells that are blank (3% missing) or exactly 10x (1% outliers); the
    appended rows are each a duplicate of some clean row (non-revenue columns
    equal, revenue related by the same blank / x10 rule).
    """
    import csv as _csv

    def read(path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as f:
            return list(_csv.DictReader(f))

    def num(raw: str) -> float | None:
        return None if raw == "" else float(raw)

    clean, dirty = read(clean_csv), read(dirty_csv)
    n = len(clean)
    other = [c for c in clean[0] if c != "revenue"] if clean else []
    out = {
        "clean_rows": n,
        "dirty_rows": len(dirty),
        "appended_rows": len(dirty) - n,
        "clean_blank_revenue_cells": sum(1 for r in clean if r["revenue"] == ""),
        "head_nonrevenue_mismatches": 0,
        "head_blank_restored": 0,
        "head_x10_restored": 0,
        "head_other_revenue_mismatches": 0,
        "head_revenue_equal": 0,
        "appended_are_duplicates_of_clean_rows": 0,
        "appended_not_matched": 0,
    }

    def relation(d: str, c: str) -> str:
        dv, cv = num(d), num(c)
        if d == c:
            return "equal"
        if dv is None and cv is not None:
            return "blank"
        if dv is not None and cv is not None and dv == cv * 10:
            return "x10"
        return "bad"

    for i in range(min(n, len(dirty))):
        d, c = dirty[i], clean[i]
        if any(d[k] != c[k] for k in other):
            out["head_nonrevenue_mismatches"] += 1
            continue
        rel = relation(d["revenue"], c["revenue"])
        if rel == "equal":
            out["head_revenue_equal"] += 1
        elif rel == "blank":
            out["head_blank_restored"] += 1
        elif rel == "x10":
            out["head_x10_restored"] += 1
        else:
            out["head_other_revenue_mismatches"] += 1

    by_key: dict[tuple, list[dict]] = {}
    for c in clean:
        by_key.setdefault(tuple(c[k] for k in other), []).append(c)
    for d in dirty[n:]:
        cands = by_key.get(tuple(d[k] for k in other), [])
        if any(relation(d["revenue"], c["revenue"]) != "bad" for c in cands):
            out["appended_are_duplicates_of_clean_rows"] += 1
        else:
            out["appended_not_matched"] += 1
    out["consistent"] = (
        out["head_nonrevenue_mismatches"] == 0
        and out["head_other_revenue_mismatches"] == 0
        and out["appended_not_matched"] == 0
    )
    return out


def render(
    mode: str,
    out_dir: Path,
    maeda_scripts: Path | None = None,
    *,
    with_clean: bool = False,
    maeda_repo: Path = DEFAULT_MAEDA_REPO,
) -> dict:
    import tempfile

    out_dir = out_dir.resolve()  # the renderer runs with cwd inside out_dir
    renderer_tmp = None
    renderer_ref = None
    renderer_commit = None
    if maeda_scripts is None:
        renderer_tmp = Path(tempfile.mkdtemp(prefix="g3_renderer_"))
        maeda_scripts, renderer_commit = materialize_pinned_renderer(
            maeda_repo, PINNED_RENDERER_REF, PINNED_RENDERER_SHA256, renderer_tmp
        )
        renderer_ref = PINNED_RENDERER_REF
    refuse_forbidden_out_dir(out_dir)
    _, public = load_event_script_source(reason="render_v2.py needs every event to render the sales data")
    _, sealed = load_sealed_source(reason="render_v2.py renders the sealed test events into the data")
    events = assemble_events(mode, public.get("events", []), (sealed or {}).get("events", []))
    frozen = _check_against_manifest(public, sealed) if mode == "v2" else None

    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(out_dir)  # generate_demo_data writes to the relative path data/demo
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(maeda_scripts))
    try:
        gen = importlib.import_module("generate_demo_data")
        gen.load_frozen_events = lambda: events
        captured: dict = {}
        if with_clean:
            # The quality-noise block starts with df.sample(frac=0.03, random_state=1)
            # on the untouched event-bearing table; take a copy there. The renderer
            # file itself is not modified, so its sha256 stays the pinned one.
            import pandas as pd

            original_sample = pd.DataFrame.sample

            def spy_sample(self, *a, **kw):
                if kw.get("random_state") == 1 and "clean" not in captured:
                    captured["clean"] = self.copy()
                return original_sample(self, *a, **kw)

            pd.DataFrame.sample = spy_sample
        try:
            gen.generate_sales()
        finally:
            if with_clean:
                pd.DataFrame.sample = original_sample
        if with_clean:
            clean_df = captured["clean"]
            clean_df.to_csv(out_dir / "sales_data_clean.csv", index=False)
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
    if renderer_ref is None:
        renderer_commit = _git_head(maeda_scripts.parent)
    receipt = {
        "mode": mode,
        "events_rendered": len(events),
        "csv_sha256": sha256_file(target),
        "csv_rows": sum(1 for _ in target.open("rb")) - 1,
        "renderer_sha256": sha256_file(renderer),
        "renderer_maeda_head": renderer_commit,
    }
    if renderer_ref:
        receipt["renderer_ref"] = renderer_ref
    if frozen:
        receipt["frozen_script"] = frozen
    name = "render_receipt.json"
    if with_clean:
        clean_path = out_dir / "sales_data_clean.csv"
        receipt = {
            "kind": "clean base table: same renderer and events as csv_sha256, quality-noise block skipped",
            **receipt,
            "dirty_csv_sha256": receipt.pop("csv_sha256"),
            "dirty_csv_rows": receipt.pop("csv_rows"),
            "clean_csv": "sales_data_clean.csv",
            "clean_csv_sha256": sha256_file(clean_path),
            "clean_csv_rows": sum(1 for _ in clean_path.open("rb")) - 1,
            "consistency": check_clean_consistency(clean_path, target),
        }
        name = "render_receipt_clean.json"
    (out_dir / name).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if renderer_tmp is not None:
        import shutil

        shutil.rmtree(renderer_tmp, ignore_errors=True)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["v1-only", "v2", "public", "baseline"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--maeda-scripts", type=Path, default=None,
                    help="override: use this directory's generate_demo_data.py instead of the pinned "
                         f"{PINNED_RENDERER_REF} version read out of MAEDA's git history")
    ap.add_argument("--with-clean", action="store_true",
                    help="also write sales_data_clean.csv (quality-noise block skipped, all events kept)")
    args = ap.parse_args()
    print(json.dumps(render(args.mode, args.out_dir, args.maeda_scripts, with_clean=args.with_clean), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
