"""Tests for the OpenMP crash fix in rag/__init__.py.

See ECOSYSTEM_INTEGRATION_PLAN.md 附录 M.4: importing faiss and voyageai (or
torch, via the crossencoder reranker) in one process, in the wrong order,
raises OMP: Error #15 and kills the whole MCP server process. The fix sets
KMP_DUPLICATE_LIB_OK before any of those native libraries can be imported —
these tests run in a subprocess so the env var starts genuinely unset,
which importing `rag` inside this same test process (already imported by
the time pytest collected this file) would not exercise.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_importing_rag_sets_kmp_duplicate_lib_ok_when_unset():
    env = {k: v for k, v in os.environ.items() if k != "KMP_DUPLICATE_LIB_OK"}
    result = subprocess.run(
        [sys.executable, "-c", "import rag, os; print(os.environ.get('KMP_DUPLICATE_LIB_OK'))"],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "TRUE"


def test_importing_rag_does_not_override_an_explicit_value():
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "FALSE"}
    result = subprocess.run(
        [sys.executable, "-c", "import rag, os; print(os.environ.get('KMP_DUPLICATE_LIB_OK'))"],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "FALSE"
