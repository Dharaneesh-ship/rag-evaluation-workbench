"""stdio safety for the MCP server.

MCP over stdio reserves stdout exclusively for the JSON-RPC protocol
stream — any stray text written to it (a debug print(), a verbose
dependency) corrupts the protocol from the client's perspective.

Two layers of protection, tested here:
1. _harden_stdio() makes the "logging/print go to stderr" invariant
   explicit and structural rather than incidental (see its docstring in
   rag/app/mcp_server/server.py for why "nobody configured stdout" isn't
   good enough on its own).
2. A static guard: no bare print() call anywhere in the shared rag/
   package (excluding rag/cli/ and rag/app/studio/, which are legitimate
   stdout-writing human-facing surfaces) — this codebase makes extensive
   use of lazy imports inside functions, so the MCP server's *reachable*
   dependency closure is effectively "all of rag/ once any tool actually
   runs," not just what's imported at module load time.
"""

from __future__ import annotations

import ast
import builtins
import io
import logging
import sys
from pathlib import Path

import pytest

_RAG_ROOT = Path(__file__).resolve().parent.parent / "rag"
_EXEMPT_DIRS = (_RAG_ROOT / "cli", _RAG_ROOT / "app" / "studio")


def _real_print_lines(path: Path) -> list[int]:
    """Line numbers of actual print(...) calls — not docstring examples."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def _is_exempt(path: Path) -> bool:
    return any(exempt in path.parents for exempt in _EXEMPT_DIRS)


class TestNoPrintOutsideCliAndStudio:
    def test_shared_rag_package_has_no_print_calls(self):
        offenders = {}
        for py_file in _RAG_ROOT.rglob("*.py"):
            if _is_exempt(py_file) or "__pycache__" in py_file.parts:
                continue
            lines = _real_print_lines(py_file)
            if lines:
                offenders[str(py_file.relative_to(_RAG_ROOT.parent))] = lines

        assert not offenders, (
            "Found print() call(s) in shared rag/ code (outside rag/cli/ and "
            "rag/app/studio/) — this can corrupt the MCP server's stdout "
            "JSON-RPC stream if this module is ever imported in that "
            f"process (directly or via a lazy import): {offenders}"
        )


class TestHardenStdio:
    @pytest.fixture(autouse=True)
    def _restore_globals(self):
        # _harden_stdio() mutates process-global state (builtins.print, the
        # root logger's handlers) — every test here must restore both, or
        # it silently breaks capsys/logging behavior for every later test
        # in the same pytest process.
        original_print = builtins.print
        original_handlers = logging.getLogger().handlers[:]
        original_level = logging.getLogger().level
        yield
        builtins.print = original_print
        logging.getLogger().handlers = original_handlers
        logging.getLogger().setLevel(original_level)

    def test_root_logger_routes_to_stderr(self):
        from rag.app.mcp_server.server import _harden_stdio

        _harden_stdio()
        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)
        assert handlers[0].stream is sys.stderr

    def test_print_is_redirected_to_stderr(self):
        from rag.app.mcp_server.server import _harden_stdio

        _harden_stdio()
        fake_out, fake_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fake_out, fake_err
        try:
            print("hello")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        assert fake_out.getvalue() == ""
        assert "hello" in fake_err.getvalue()

    def test_print_with_explicit_file_kwarg_still_redirected(self):
        # A caller passing file=<some stream> explicitly must not crash (a
        # naive functools.partial(print, file=sys.stderr) would raise
        # "got multiple values for keyword argument 'file'") and the output
        # must still land on stderr, not the explicitly requested stream.
        from rag.app.mcp_server.server import _harden_stdio

        _harden_stdio()
        requested_stream = io.StringIO()
        real_stderr = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = real_stderr
        try:
            print("explicit-file-request", file=requested_stream)  # must not raise
        finally:
            sys.stderr = old_stderr
        assert "explicit-file-request" not in requested_stream.getvalue()
        assert "explicit-file-request" in real_stderr.getvalue()
