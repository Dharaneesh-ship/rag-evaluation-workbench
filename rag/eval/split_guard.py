"""Test-split reveal guard — the single mechanism, not one entry point's check.

硬约束 1: the TB5 test split gets exactly ONE official reveal. 附录 CI.4 built
the first version of this guard (``filter_by_split`` /
``warn_if_test_split_involved``) and wired it into ``rag.cli.eval``. 附录 CP.4
then showed why that was not enough:

    grep -rn "filter_by_split" --include=*.py .  ->  rag/cli/eval.py only

Both probe scripts written since (附录 CJ.3, 附录 CO) read
``eval/fixtures/event_script/event_script.yaml`` directly with
``yaml.safe_load`` and never touched ``rag.cli.eval`` at all — so they never
touched the guard either. 附录 CO.0 is a recorded, self-reported test-split
observation that walked straight through that hole. The hand-written
``assert split == "dev"`` in the CO probe script was personal discipline; the
next script does not inherit it.

Two things follow, and this module implements both.

1.  **Loading must default to dev and make anything else declare itself.**
    ``load_event_script()`` with no argument yields dev-split events only.
    Test-split content requires passing ``split="test"``/``"all"``
    explicitly, and prints the reveal banner when it does.

2.  **The load path can always be bypassed — ``yaml.safe_load`` needs no
    cooperation from us — so the guard with actual teeth sits at
    *execution*, not at load.** ``assert_not_test_probe_query()`` recognises
    a frozen test-split probe query *by its content* and refuses it unless
    a reveal has been declared. It is called from
    ``QueryPipeline.query()`` (the one funnel every retrieval in this
    repository passes through: CLI eval, the MCP ``retrieve`` tools,
    ``rag.cli.query``, and any ad-hoc script that imports any of them) and
    again, earlier and more cheaply, from ``rag.cli.eval._run_queries``.

The division of labour is deliberate: **handling test-split text is loud,
executing it is blocked.** Writing ``eval/fixtures/tb5_retrieval/test.json``
(``build_suite.py``) or hashing the frozen YAML (``verify.py``) has to read
every split and is not a reveal — 附录 CR.5 already ruled that reading
structural metadata does not consume the one reveal. Running retrieval over a
test-split probe query is the irreversible act, and that is what fails closed.

Recognition is content-based, per 附录 CK.6: the guard matches the normalised
*text* of a query against digests of the frozen test-split probe queries. It
therefore does not care which file the caller read, what that file was named,
whether it was copied, or whether the caller went through
``event_script.yaml``, through the derived ``test.json``, or typed the string
in by hand. It also means this module holds only SHA-256 digests at rest: it
can recognise a test-split probe query without exposing one, and the error it
raises never echoes the query text back into a log or traceback.
"""

from __future__ import annotations

import hashlib
import os
import sys
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]

#: The frozen event script (git tag ``event-script-v1``). Absent in a packaged
#: deployment that does not ship ``eval/`` — see ``_load_document``.
EVENT_SCRIPT_PATH = _ROOT / "eval" / "fixtures" / "event_script" / "event_script.yaml"

#: v2 (附录 FR.4): the v2 test events live in a local-only, gitignored sealed
#: file. It is never part of any tag and never pushed. Absent on a fresh clone,
#: which is why the guard's recognition does not depend on it: see
#: ``PROBE_DIGEST_LIST_PATH``.
SEALED_SCRIPT_PATH = _ROOT / "eval" / "sealed" / "event_script_v2_test.yaml"

#: Hash-only list of every test-split probe digest (``probe_queries`` and
#: ``h2_probes``, public and sealed). Committed; holds no query text.
PROBE_DIGEST_LIST_PATH = EVENT_SCRIPT_PATH.parent / "test_probe_digests.json"

V1_TAG = "event-script-v1"

SPLIT_DEV = "dev"
SPLIT_TEST = "test"
SPLIT_ALL = "all"
VALID_SPLIT_SELECTORS = frozenset({SPLIT_DEV, SPLIT_TEST, SPLIT_ALL})

#: Process-level reveal declaration. Set it to a non-empty, non-``"0"`` value
#: to allow test-split probe queries to execute in this process. Intended for
#: the one predetermined A/B reveal run, including one driven from outside
#: this repository (MAEDA's stage-4 harness).
REVEAL_ENV_VAR = "RAG_TEST_SPLIT_REVEAL"


class TestSplitRevealError(RuntimeError):
    """Raised when a test-split probe query is about to be executed undeclared.

    The message deliberately does not contain the offending query text: the
    whole point of the guard is that this string should not be spread further
    (into a traceback, a CI log, a pytest report) by the act of blocking it.
    """


# ---------------------------------------------------------------------------
# Loading the frozen event script
# ---------------------------------------------------------------------------

_document_cache: dict[Path, dict[str, Any]] = {}
_digest_cache: dict[Path, frozenset[str]] = {}


def _resolve_path(path: Path | str | None) -> Path:
    return Path(path) if path is not None else EVENT_SCRIPT_PATH


def _load_document(path: Path) -> dict[str, Any]:
    """Parse the event script, caching per path. ``{}`` when the file is absent.

    A missing file is not an error: the fixture only exists in a source
    checkout. In a deployment that does not ship ``eval/`` there is no test
    split to protect, so the guard degrades to a no-op there rather than
    breaking imports. On the machines that *do* hold the frozen script — the
    ones where 硬约束 1 is live — it is always present.
    """
    resolved = path.resolve()
    cached = _document_cache.get(resolved)
    if cached is not None:
        return cached
    if not resolved.is_file():
        _document_cache[resolved] = {}
        return {}
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    _document_cache[resolved] = data
    return data


def clear_cache() -> None:
    """Drop the parsed-document and digest caches (tests that patch the path).

    Also re-arms the once-per-process environment-variable reveal banner, so a
    test that exercises that path does not silence it for the rest of the
    session.
    """
    global _env_reveal_announced
    _document_cache.clear()
    _digest_cache.clear()
    _env_reveal_announced = False


def load_event_script_source(
    *, reason: str, path: Path | str | None = None
) -> tuple[str, dict[str, Any]]:
    """Return ``(raw text, parsed document)`` for the whole event script.

    The raw text is needed by ``verify.py`` for byte-level hygiene checks (BOM,
    CRLF, trailing newline) and by nothing else. Same declaration contract as
    :func:`load_event_script_document`.

    Raises:
        ValueError: If ``reason`` is empty or blank.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "reading the full event script requires a non-empty reason: reading "
            "every split has to be an explicit, attributable act."
        )
    resolved = _resolve_path(path).resolve()
    text = resolved.read_text(encoding="utf-8") if resolved.is_file() else ""
    data = _load_document(resolved)
    if text or data:
        _print_full_document_banner(reason)
    return text, data


def load_event_script_document(*, reason: str, path: Path | str | None = None) -> dict[str, Any]:
    """Return the whole parsed event script, including every split.

    For the two in-repository tools that legitimately need all splits:
    ``eval/fixtures/event_script/verify.py`` (structural verification and the
    manifest hash, which is computed over the whole document) and
    ``eval/fixtures/tb5_retrieval/build_suite.py`` (which writes both
    ``dev.json`` and ``test.json``).

    ``reason`` is required and must be non-empty — this is the declaration.
    Nothing here consumes the one reveal: producing or hashing test-split
    *text* is not observing a test-split *result* (附录 CR.5). Executing one
    of these queries is what fails closed, in
    ``assert_not_test_probe_query()``.

    Args:
        reason: Why this caller needs all splits. Printed with the banner.
        path: Override the event-script path (tests).

    Returns:
        The parsed YAML document, or ``{}`` if the fixture is absent.

    Raises:
        ValueError: If ``reason`` is empty or blank.
    """
    return load_event_script_source(reason=reason, path=path)[1]


def load_sealed_source(
    *, reason: str, path: Path | str | None = None
) -> tuple[str, dict[str, Any]]:
    """Return ``(raw text, parsed document)`` of the sealed v2 test file.

    ``("", {})`` when the file is absent (fresh clone, packaged deployment).
    Same declaration contract as :func:`load_event_script_source`; the only
    in-repository readers are ``verify.py`` and the rag-side render script,
    both of which legitimately need the test events to hash and render them.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "reading the sealed v2 test file requires a non-empty reason: "
            "reading every split has to be an explicit, attributable act."
        )
    resolved = (Path(path) if path is not None else SEALED_SCRIPT_PATH).resolve()
    text = resolved.read_text(encoding="utf-8") if resolved.is_file() else ""
    data = _load_document(resolved)
    if text or data:
        _print_full_document_banner(reason)
    return text, data


def load_v1_tag_events(*, tag: str = V1_TAG, repo: Path | str | None = None) -> list[dict[str, Any]]:
    """Return the event list exactly as frozen at git tag ``event-script-v1``.

    v2 inherits every v1 split from this tag (附录 FR.1 point 1) instead of
    keeping a second copy of the mapping in code. Uses ``git show``; raises
    ``RuntimeError`` when the tag or git is unavailable.
    """
    import subprocess

    rel = EVENT_SCRIPT_PATH.relative_to(_ROOT).as_posix()
    result = subprocess.run(
        ["git", "-C", str(repo or _ROOT), "show", f"{tag}:{rel}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot read tag {tag!r}: {result.stderr.strip()}")
    return (yaml.safe_load(result.stdout) or {}).get("events", [])


def load_event_script(
    split: str = SPLIT_DEV,
    *,
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return event entries for ``split``. Default ``dev`` — safe by default.

    This is the sanctioned way to read events out of the frozen script. Test
    content is not reachable without naming it: ``split`` defaults to
    ``"dev"``, and ``"test"``/``"all"`` print the reveal banner.

    Args:
        split: ``"dev"`` (default), ``"test"``, or ``"all"``.
        path: Override the event-script path (tests).

    Returns:
        Shallow-copied event dicts for the requested split, in file order.

    Raises:
        ValueError: If ``split`` is not one of the three selectors.
    """
    if split not in VALID_SPLIT_SELECTORS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLIT_SELECTORS)}, got {split!r}")
    data = _load_document(_resolve_path(path))
    events = [dict(e) for e in data.get("events", [])]
    selected = filter_by_split(events, split)
    warn_if_test_split_involved(selected)
    return selected


def load_event_metadata(*, path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return every event with ``probe_queries`` replaced by count + digests.

    附录 CR.5 ruled that reading structural metadata — which events exist,
    which split each is in — does not consume the one reveal, and the guard
    itself could not work otherwise. This accessor makes that the easy path:
    it hands back split labels, event types and probe *counts* for all 12
    events without ever handing back a test-split probe query string.
    """
    data = _load_document(_resolve_path(path))
    out: list[dict[str, Any]] = []
    for event in data.get("events", []):
        redacted = dict(event)
        probes = [p for p in (event.get("probe_queries") or []) if isinstance(p, str)]
        redacted.pop("probe_queries", None)
        redacted.pop("h2_probes", None)
        redacted["probe_query_count"] = len(probes)
        redacted["probe_query_digests"] = [digest_of(p) for p in probes]
        out.append(redacted)
    return out


def iter_probe_queries(
    split: str = SPLIT_DEV,
    *,
    path: Path | str | None = None,
) -> Iterator[tuple[str, int, str]]:
    """Yield ``(event_id, 1-based index, query text)`` for ``split``.

    Same default-safe contract as :func:`load_event_script`. This is what a
    probe script should call instead of ``yaml.safe_load`` — 附录 CJ.3 and
    附录 CO both hand-rolled this loop, and 附录 CO.0 is what that cost.
    """
    for event in load_event_script(split, path=path):
        event_id = event.get("event_id", "?")
        for i, query in enumerate(event.get("probe_queries") or [], start=1):
            yield event_id, i, query


# ---------------------------------------------------------------------------
# Split-aware case selection (附录 CI.4) — moved here from rag/cli/eval.py
# ---------------------------------------------------------------------------
#
# 附录 CH.6/CI.4: before this existed, eval/fixtures/tb5_retrieval/test.json
# was a plain, unprotected JSON file — `rag.cli.eval --suite-file
# .../test.json` could be run by anyone/any agent with zero warning and zero
# refusal, the only "protection" being a docstring sentence in
# build_suite.py ("FROZEN — only run at the final A/B reveal") that no code
# path ever checked. That is structurally the same gap 附录 CC.1 found and
# fixed in MAEDA's scripts/measure_noise.py (--split, default dev). This
# mirrors that fix, but keyed off each entry's own 'split' field (added by
# build_suite.py, 附录 CI.4) rather than off a filename/path guess — a
# content-based check survives copies, renames, or a --suite-file argument
# aimed straight at test.json, which is exactly the "bypass the split
# selector, name the file/cases directly" path CI.4 requires covering (see
# MAEDA's test_warns_even_for_explicit_cases_that_happen_to_include_test).
#
# 附录 CP.4 moved both functions out of rag/cli/eval.py and into this module,
# unchanged in behaviour, so that they stop being "the check rag.cli.eval
# happens to run" and become the shared vocabulary every reader of the event
# script and its derived suites uses. rag.cli.eval re-exports them.


def filter_by_split(entries: list[dict], split: str) -> list[dict]:
    """Filter suite entries by their self-declared ``split`` field.

    Entries produced by non-TB5 fixtures (``example_queries.json``,
    ``failure_cases.json``) have no ``split`` field at all and are never
    treated as test-split, so this filter is a no-op for them regardless of
    the requested split — only ``eval/fixtures/tb5_retrieval/{dev,test}.json``
    carry the marker.

    Args:
        entries: Suite entries as loaded from JSON/JSONL.
        split: ``"dev"`` (default) excludes every ``split == "test"`` entry;
            ``"test"`` keeps only ``split == "test"`` entries; ``"all"``
            returns everything unfiltered.

    Returns:
        The filtered entry list. Order is preserved.
    """
    if split == "all":
        return list(entries)
    if split == "test":
        return [e for e in entries if e.get("split") == "test"]
    return [e for e in entries if e.get("split") != "test"]


def _emit_banner(lines: list[str]) -> None:
    """Write a banner to **stderr**, deliberately not through ``print()``.

    附录 CP.4/CZ. The banners exist for 硬约束 1: a human launching an eval
    run must see, on the terminal they launched it from, that test-split
    content is involved. The obvious way to do that is ``print()``, and that
    is what 附录 CI.4 did while these functions still lived in
    ``rag/cli/eval.py`` — legitimately, because ``rag/cli/`` is a human-facing
    stdout surface.

    Moving the guard into the shared ``rag/`` package changed which
    invariants apply to it. ``QueryPipeline.query()`` imports this module, so
    the MCP stdio server's reachable closure now contains it, and stdout there
    is reserved for the JSON-RPC stream (``tests/test_mcp_stdio_safety.py``,
    ``rag/app/mcp_server/server.py::_harden_stdio``). Writing to ``sys.stderr``
    directly satisfies both: an operator at a terminal sees stderr exactly as
    they see stdout, and the protocol stream is untouched by construction
    rather than by remembering to call ``_harden_stdio()`` first. It is the
    same reasoning ``retrieval_config.log_server_default_retrieval_tier``
    already records for its startup log.

    Not ``print(..., file=sys.stderr)``: the static guard matches ``print``
    calls by AST regardless of the ``file=`` keyword, and the point here is to
    keep that invariant honest rather than to satisfy it on a technicality.

    Flushed explicitly — stderr is unbuffered on CPython today, but this
    banner's whole job is to appear *before* the thing it warns about.
    """
    sys.stderr.write("\n" + "!" * 78 + "\n")
    for line in lines:
        sys.stderr.write(f"!! {line}\n")
    sys.stderr.write("!" * 78 + "\n\n")
    sys.stderr.flush()


def warn_if_test_split_involved(selected: list[dict]) -> None:
    """Emit an unmissable warning banner if any selected entry is test-split.

    Fires off the actually-selected set, not off which ``--split`` value
    was passed — so an explicit ``--suite-file`` pointed at a file that
    happens to contain test-marked entries is flagged exactly the same as
    ``--split test``/``--split all`` (附录 CI.4's bypass-coverage
    requirement). Written to stderr rather than merely logged, per 硬约束 1:
    the test split gets exactly ONE official reveal, and this is the last
    point before a real eval run actually executes where a human is still
    watching the terminal it was launched from. See :func:`_emit_banner` for
    why stderr and not stdout.

    Args:
        selected: The entries about to actually be evaluated (post-filter).
    """
    test_ids = sorted(e.get("query_id", e.get("query", "?")) for e in selected if e.get("split") == "test")
    if not test_ids:
        return
    _emit_banner(
        [
            "WARNING: this eval run includes TEST-SPLIT queries (TB5 retrieval-layer eval).",
            "硬约束 1: test split gets exactly ONE official reveal --",
            "if this is that reveal, make sure it's the real predetermined",
            "run, not routine dev-split tuning or exploration.",
            f"{len(test_ids)} test-split quer(ies): {test_ids}",
            "Routine tuning should use --split dev (the default) instead.",
        ]
    )


def _print_full_document_banner(reason: str) -> None:
    _emit_banner(
        [
            "NOTE: reading the FULL frozen event script, every split included.",
            f"Declared reason: {reason}",
            "硬约束 1 / 附录 CR.5: producing or hashing test-split text is not",
            "a reveal. RUNNING retrieval over it is, and that is blocked",
            "separately (rag.eval.split_guard.assert_not_test_probe_query).",
        ]
    )


def _print_reveal_banner(reason: str) -> None:
    _emit_banner(
        [
            "TEST-SPLIT REVEAL DECLARED — test-split probe queries may now execute.",
            f"Declared reason: {reason}",
            "硬约束 1: the test split gets exactly ONE official reveal. If this",
            "is not the predetermined run, abort now: it is not undoable.",
        ]
    )


# ---------------------------------------------------------------------------
# Content recognition — digests, never the strings themselves
# ---------------------------------------------------------------------------


def normalize_query(text: str) -> str:
    """Normalise a query for content matching: NFKC, collapse space, casefold.

    Deliberately forgiving about whitespace and case so that a copy-pasted or
    re-wrapped probe query is still recognised. It is *not* trying to defeat a
    determined adversary — nobody here is adversarial, and someone who edits a
    query to slip past a digest has by definition declared their intent to
    themselves. The failure mode this guard exists to prevent is the accidental
    one: 附录 CO.0, where the entries simply came in through a door with no
    check on it.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def digest_of(text: str) -> str:
    """SHA-256 of the normalised query text."""
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()


def probe_texts_of(event: dict[str, Any]) -> list[str]:
    """Every string in an event that is a runnable probe (both fields)."""
    out: list[str] = []
    for field in ("probe_queries", "h2_probes"):
        out.extend(q for q in (event.get(field) or []) if isinstance(q, str))
    return out


def build_probe_digest_list(
    public: dict[str, Any], sealed: dict[str, Any] | None = None
) -> list[str]:
    """Sorted digests of all test-split probes: public test events, sealed v2
    test events, and the sealed ``h2_overlay`` (h2_probes of v1 test events)."""
    digests: set[str] = set()
    for event in public.get("events", []):
        if event.get("split") == SPLIT_TEST:
            digests.update(digest_of(q) for q in probe_texts_of(event))
    sealed = sealed or {}
    for event in sealed.get("events", []):
        if event.get("split") == SPLIT_TEST:
            digests.update(digest_of(q) for q in probe_texts_of(event))
    for queries in (sealed.get("h2_overlay") or {}).values():
        digests.update(digest_of(q) for q in queries if isinstance(q, str))
    return sorted(digests)


def test_probe_query_digests(*, path: Path | str | None = None) -> frozenset[str]:
    """Digests of every test-split probe query.

    Union of (a) digests computed from the public event script's test-split
    events and (b) the committed hash-only list next to it, which also covers
    the sealed v2 test events (so the guard bites on a machine that does not
    hold the sealed file). Cached per path. This is the only place in the
    running process that ever materialises test-split probe text, and it keeps
    none of it.
    """
    resolved = _resolve_path(path).resolve()
    cached = _digest_cache.get(resolved)
    if cached is not None:
        return cached
    data = _load_document(resolved)
    digests = {
        digest_of(query)
        for event in data.get("events", [])
        if event.get("split") == SPLIT_TEST
        for query in probe_texts_of(event)
    }
    list_path = resolved.parent / PROBE_DIGEST_LIST_PATH.name
    if list_path.is_file():
        import json as _json

        listed = _json.loads(list_path.read_text(encoding="utf-8")).get("digests", [])
        digests.update(d for d in listed if isinstance(d, str))
    frozen = frozenset(digests)
    _digest_cache[resolved] = frozen
    return frozen


def is_test_probe_query(text: str, *, path: Path | str | None = None) -> bool:
    """True if ``text`` is (a normalised match for) a test-split probe query."""
    if not isinstance(text, str) or not text.strip():
        return False
    return digest_of(text) in test_probe_query_digests(path=path)


# ---------------------------------------------------------------------------
# The declaration, and the gate it opens
# ---------------------------------------------------------------------------

_declared_reason: str | None = None
_env_reveal_announced = False


def reveal_declared() -> str | None:
    """Return the active reveal declaration, or ``None``.

    Two ways to declare, both explicit: the :data:`REVEAL_ENV_VAR`
    environment variable (for a run driven from outside this process, e.g.
    MAEDA's stage-4 harness) or :func:`declare_test_split_reveal` (in-process,
    scoped, used by ``rag.cli.eval --split test``).

    附录 CZ: the environment-variable door announces itself the first time it
    is actually used, once per process. Without that it was the *quietest* way
    to open the gate — and it is also the easiest one to open by accident,
    because several entry points call ``load_dotenv(override=False)``
    (``rag/cli/ingest.py``, the Studio pages, ``scripts/tb5_build_corpus_index``),
    so a line left in a ``.env`` file becomes an ambient, invisible reveal
    permission. The in-process contextmanager already printed a banner; a
    silent alternative route around a banner is not a guard.
    """
    global _env_reveal_announced
    if _declared_reason is not None:
        return _declared_reason
    env = os.environ.get(REVEAL_ENV_VAR, "").strip()
    if env and env != "0":
        reason = f"{REVEAL_ENV_VAR}={env}"
        if not _env_reveal_announced:
            _env_reveal_announced = True
            _print_reveal_banner(f"{reason} (from the environment)")
        return reason
    return None


@contextmanager
def declare_test_split_reveal(reason: str) -> Iterator[None]:
    """Declare, for the duration of the block, that this is the reveal run.

    Prints the reveal banner on entry. Nested declarations are allowed; the
    outermost one wins, and the state is restored on exit even if the block
    raises.

    Args:
        reason: Why this is the predetermined reveal. Required, non-empty.

    Raises:
        ValueError: If ``reason`` is empty or blank.
    """
    global _declared_reason
    if not reason or not reason.strip():
        raise ValueError(
            "declare_test_split_reveal() requires a non-empty reason: 硬约束 1's "
            "one reveal has to be attributable to a stated purpose."
        )
    previous = _declared_reason
    _declared_reason = reason
    _print_reveal_banner(reason)
    try:
        yield
    finally:
        _declared_reason = previous


def assert_not_test_probe_query(
    texts: str | Iterable[str],
    *,
    context: str = "",
    path: Path | str | None = None,
) -> None:
    """Refuse to let a frozen test-split probe query execute undeclared.

    This is the mechanism 附录 CP.4 asked for. It does not care how the caller
    obtained the string — ``load_event_script``, raw ``yaml.safe_load`` over
    the frozen YAML, ``json.load`` over the derived ``test.json``, or a
    literal pasted into a scratch script. It matches on content (附录 CK.6),
    fails closed, and is opened only by an explicit declaration
    (:func:`declare_test_split_reveal` or :data:`REVEAL_ENV_VAR`).

    Args:
        texts: A query string, or an iterable of them.
        context: Short description of the call site, for the error message.
        path: Override the event-script path (tests).

    Raises:
        TestSplitRevealError: If any input is a test-split probe query and no
            reveal has been declared. The message names the offending entries
            by digest prefix only, never by text.
    """
    if isinstance(texts, str):
        texts = [texts]
    offenders = [t for t in texts if is_test_probe_query(t, path=path)]
    if not offenders:
        return
    if reveal_declared():
        return
    where = f" ({context})" if context else ""
    raise TestSplitRevealError(
        f"Refusing to run retrieval over {len(offenders)} frozen TEST-SPLIT probe "
        f"quer(ies){where}: digest prefixes "
        f"{sorted(digest_of(t)[:12] for t in offenders)}.\n"
        "硬约束 1: the TB5 test split gets exactly ONE official reveal, and this "
        "process has not declared that it is that run.\n"
        "If you are exploring or tuning, use the dev split — "
        "rag.eval.split_guard.load_event_script() defaults to it, and "
        "`python -m rag.cli.eval --split dev` is the default there too.\n"
        "If this really is the predetermined reveal, declare it: wrap the call in "
        "rag.eval.split_guard.declare_test_split_reveal(reason=...) or set "
        f"{REVEAL_ENV_VAR}=1 in the environment.\n"
        "(附录 CP.4 — this check is content-based; it fires no matter which file "
        "or entry point the query came from. 附录 CO.0 is why it exists.)"
    )
