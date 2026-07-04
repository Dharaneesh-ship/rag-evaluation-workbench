# RAG Framework — Claude Code Instructions

This project uses a DEV_SPEC-driven development workflow. All implementation work follows `dev_spec_v8.md` as the single source of truth.

## Skills

This project has 3 workflow skills in `.claude/skills/` (flat `.md` files, triggered by the keywords in Quick Commands below):

- **dev-workflow** — Main orchestrator. Finds the next task, coordinates implementation and checkpoint.
- **implement** — Reads task spec, writes code, runs tests, iterates on failures.
- **checkpoint** — Summarizes completed work, updates progress in DEV_SPEC, prepares git commit.

It also has 2 standard Claude Code skills (auto-discovered):

- **resume-writer** (`.claude/skills/resume-writer/SKILL.md`) — Generates tailored resume project-experience bullets from this project's real technical highlights.
- **interview-prep** (`.claude/skills/interview-prep/SKILL.md`) — Mock technical interviewer that grills a resume against this project's real implementation (RRF/FAISS/BM25 internals, chunk_id hashing, MCP transport, etc.) to catch exaggeration; generates a scored report.

## Quick Commands

- `next task` or `下一阶段` → Run full dev-workflow pipeline (find task → implement → test → checkpoint)
- `continue` or `继续` → Resume implementation from where it left off (skip task finding)
- `status` or `进度` → Show current progress without doing any work
- `checkpoint` or `保存` → Save progress for current task only

## Key Rules

1. **DEV_SPEC is the single source of truth.** Always read `dev_spec_v8.md` before starting any task.
2. **One task at a time.** Complete and checkpoint one task before moving to the next.
3. **Never skip tests.** If a task has a test method, run it. If tests fail, fix and re-run (max 3 attempts).
4. **Ask before committing.** Always get user confirmation before git commit.
5. **All code in English.** Comments, docstrings, variable names — all English.
