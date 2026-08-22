"""Evaluation-support code that must be importable from anywhere in ``rag``.

Not to be confused with the top-level ``eval/`` directory, which holds
*fixtures* (the frozen event script, the TB5 retrieval suites and their
corpus). This package holds the *code* that enforces policy over those
fixtures — most importantly ``split_guard``, the test-split reveal guard
(附录 CI.4 / CP.4).
"""
