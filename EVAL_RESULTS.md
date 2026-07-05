# Retrieval Evaluation & Ingest Performance: ASQA Study

This document reports a retrieval evaluation built on [ASQA](https://huggingface.co/datasets/din0s/asqa)
(Ambiguous Short-Answer QA, Stelmakh et al. 2022), a human-labeled long-form QA
dataset. It answers three concrete engineering questions for this project:

1. **What chunk token budget should the chunker use?**
2. **Does hybrid (BM25 + embedding) retrieval actually beat BM25 alone, and by how much?**
3. **What are the ingest-pipeline's throughput and memory bottlenecks, and how much do the fixes help?**

Everything below is reproducible with one script (`scripts/asqa_eval.py`) and a
fixed random seed. Raw per-question results for every run are saved as JSON
under `data/eval/asqa/*.json` (not tracked in git — see
[Reproducing these results](#reproducing-these-results)).

---

## TL;DR

| Config | Recall@5 | Recall@10 | Query latency |
|---|---:|---:|---:|
| chunk=128, BM25 | 45.9% | 52.7% | 74 ms |
| chunk=256, BM25 | 51.7% | 59.0% | 43 ms |
| **chunk=512, BM25** | **61.2%** | **68.8%** | 25 ms |
| chunk=768, BM25 | 61.8% | 69.2% | 20 ms |
| **chunk=512, BM25 + OpenAI embeddings (hybrid)** | **67.8%** | **74.6%** | 301 ms |

- **512 tokens is the chunk-size sweet spot.** Recall climbs sharply from 128→512
  tokens, then flattens: 512→768 buys only +0.6pt recall@5 for a chunk count that
  barely changes (6,363 → 6,025). 128-token chunks are too fragmented — the
  question's keywords and the answer frequently end up split across chunks.
- **Hybrid retrieval adds +6.6pt recall@5 / +5.8pt recall@10** over BM25-only at
  the same chunk size, at the cost of ~12x query latency (25ms → 301ms — this
  run makes one OpenAI embedding API call per query; latency would drop
  substantially with a local embedding model or query caching).
- **A production bug was found and fixed during this study**: BM25-only ingestion
  (`embedding_provider=None`) silently never populated the keyword index. See
  [Bug found during this study](#bug-found-during-this-study).
- **Two ingest throughput bottlenecks were found and fixed**: Vision-LLM page
  transcription ran one page at a time (network-bound, embarrassingly
  parallel), and batch ingest rebuilt the entire BM25 index from scratch after
  every document (O(N²) for a batch of N). See
  [Ingest performance optimizations](#ingest-performance-optimizations).
- **A memory-scaling defect in batch ingest was found and fixed**: peak memory
  during `ingest_many()` grew with the *total batch size*, not the configured
  `max_workers` — the concurrency knob didn't actually bound memory the way
  its name implies. See [§4](#4-bounding-ingest_many-peak-memory-to-max_workers).

---

## Why ASQA, and why it needed a custom corpus

Most retrieval benchmarks (BEIR, MTEB) ship a fixed, pre-chunked corpus — every
"document" is already a short, single passage. That's convenient, but it means
chunk-size experiments are meaningless: the chunking decision has already been
made by whoever built the benchmark.

ASQA is different: it provides 948 human-labeled ambiguous questions, each with
short reference answers (`qa_pairs[].short_answers`) and the Wikipedia pages the
annotators used (`wikipages[].url`) — but **no pre-made corpus**. That's exactly
what this evaluation needs: real HTML documents that this project's own
ingest pipeline (loader → sniffer → trafilatura parser → cleaner → chunker) has
to chunk itself, so the chunk-size variable is actually under test.

The trade-off: ASQA's official metric (string-overlap F1 against long-form
generated answers) evaluates *generation* quality, not retrieval. This study
repurposes the human short-answers for a different, narrower metric — see
below.

## Methodology

### 1. Sampling

- Source: `din0s/asqa`, `dev` split (948 examples total).
- Sample: 150 examples, seed `42` (`random.Random(42).sample(...)`),
  deterministic and reproducible.
- Each ASQA example decomposes into multiple `qa_pairs` (disambiguated
  sub-questions). Filtering to sub-questions with non-empty `short_answers`
  yields **503 evaluation questions** from the 150 sampled examples.

### 2. Corpus construction

- Every `wikipages[].url` referenced by the 150 sampled examples was collected
  and deduped → **356 unique Wikipedia URLs**.
- Fetched via `trafilatura.fetch_url()` (same fetch mechanism as this project's
  `WebConnector`), cached to disk as content-addressed HTML
  (`data/eval/asqa/corpus/<sha256(url)[:20]>.html`) so repeated runs across
  chunk-size configs never re-hit Wikipedia.
- 1 of 356 URLs returned an empty response and was skipped (network/redirect
  issue, not a parser problem).
- The remaining 355 pages were ingested through the **actual production
  `IngestPipeline`** — the same code path a user's PDF/HTML/DOCX upload goes
  through (loader → `CompositeSniffer` → `trafilatura` HTML parser → cleaner →
  `AnchorAwareChunkPacker` → DocStore + BM25/FAISS index). No shortcuts or
  synthetic chunking were used.
- Of the 355 ingested pages, **350 succeeded** and **5 produced zero chunks**
  after parsing (`Sherlock (TV series)`, `Ministry of Environment, Forest and
  Climate Change`, `Svalbard Global Seed Vault`, `City of London`, `Grand
  Central Terminal` — thin/heavily-templated Wikipedia pages where trafilatura
  extracted under the 50-character quality-gate minimum). These were correctly
  flagged, removed from the store, and logged — see
  [`quality_gates.py`](rag/pipelines/parsing/quality_gates.py) — rather than
  silently indexed as empty documents.
- Each token-budget / embedding-mode combination gets its own isolated SQLite
  DocStore + BM25/FAISS index directory (`data/eval/asqa/tb{budget}[_{mode}].db`)
  so runs never interfere with each other or with production data
  (`data/rag.db` is never touched).

### 3. Retrieval metric: Recall@k (answer-substring match)

For each of the 503 evaluation questions:

1. Run the question through `QueryPipeline.query()`.
2. Take the top-*k* retrieved chunks' `display_text`.
3. **Hit** if *any* of the question's human-labeled `short_answers` strings
   appears as a case-insensitive substring anywhere in the top-*k* chunks;
   **miss** otherwise.
4. `Recall@k = hits / 503`.

This is a standard retrieval-stage proxy metric (distinct from ASQA's own
generation-quality metric) — it asks "did retrieval surface a chunk containing
the answer," not "did the system's final generated answer match a human
reference." It requires no LLM judge, costs nothing extra to compute, and is
directly interpretable: a RAG system's generator can only cite what retrieval
put in front of it, so this is a hard upper bound on end-to-end answer
correctness.

**Caveat**: substring matching has known false positives/negatives (e.g. a
chunk mentioning "2006" for an unrelated year would count as a hit for a
question whose answer is "2006"). At n=503 questions this noise is expected to
mostly wash out across configs, but absolute recall numbers should be read as
directional, not exact.

---

## Results

### Chunk size sweep (BM25-only, embedding=none)

| Token budget | Recall@5 | Recall@10 | Chunks (350 docs) | Avg chunks/doc | Query latency |
|---:|---:|---:|---:|---:|---:|
| 128 | 45.9% | 52.7% | 25,274 | 72.2 | 74 ms/query |
| 256 | 51.7% | 59.0% | 14,116 | 40.3 | 43 ms/query |
| **512** | **61.2%** | **68.8%** | 6,363 | 18.2 | 25 ms/query |
| 768 | 61.8% | 69.2% | 6,025 | 17.2 | 20 ms/query |

**Reading**: recall improves monotonically with chunk size but with sharply
diminishing returns. 128→512 tokens gains **+15.3pt** recall@5; 512→768 gains
only **+0.6pt** for a chunk count reduction of just 5.3% (6,363 → 6,025) — most
Wikipedia sections are already shorter than 512 tokens, so a larger budget
mostly stops mattering. **512 tokens is the recommended production default**:
it captures almost all of the achievable recall while keeping chunk count
(and, in production, embedding cost) close to 3.6x lower than the 128-token
config.

Why smaller chunks lose recall: BM25 scores on lexical (keyword) overlap
between the query and a single chunk. When a chunk is too small, the
question's keywords and the sentence containing the answer are more likely
to land in *different* chunks — neither one scores well alone. Larger chunks
are more likely to contain both the query's topic *and* the specific answer
sentence together.

### Hybrid retrieval: BM25-only vs. BM25 + OpenAI embeddings (chunk=512)

| Config | Recall@5 | Recall@10 | Query latency |
|---|---:|---:|---:|
| BM25-only | 61.2% | 68.8% | 25 ms/query |
| BM25 + OpenAI `text-embedding-3-small` (RRF fusion) | **67.8%** | **74.6%** | 301 ms/query |
| **Δ (hybrid − BM25)** | **+6.6pt** | **+5.8pt** | 12.0x |

**Reading**: semantic retrieval closes a real gap that pure keyword matching
cannot. Many ASQA questions phrase the query differently from how the answer
page states the fact (e.g. "who has scored the most international goals" vs.
a page that states the player's name and goal count without repeating the
question's phrasing) — BM25 has no way to match these; a dense embedding
captures the semantic relationship anyway.

The cost is real too: latency goes from 25ms to 301ms per query (roughly the
network round-trip for one OpenAI embedding call per query in this
evaluation harness). In a production system this would typically be
mitigated by caching frequent queries, batching, or switching to a
locally-hosted embedding model (this project also ships a `multilingual`
local-embedding option — see `rag/infra/embedding/multilingual_embedding.py`)
— but the raw trade-off is worth stating plainly rather than hiding it.

---

## Bug found during this study

Building the BM25-only pilot run initially returned **0% recall** across all
503 questions — every query returned zero candidates. Root-causing this (with
a minimal, ASQA-independent repro: ingest one plain `.txt` file with
`embedding_provider=None`) found a real defect in
[`rag/pipelines/ingest_pipeline.py`](rag/pipelines/ingest_pipeline.py):

```python
# Before (bug): keyword_index.add() was nested inside the embedding block,
# so BM25-only ingestion (embedding_provider=None) never indexed anything.
if self._embedding_provider is not None and chunks:
    ...
    if self._keyword_index is not None:
        self._keyword_index.add(chunks)   # never reached when embedding_provider is None
```

`IngestResult.status` still reported `"ok"` and `chunk_count` was correct —
chunks were persisted to the DocStore — but the BM25 keyword index was left
completely empty, silently. This affected **any BM25-only configuration**,
including the Studio UI's "Provider: none — Skip embedding (BM25 only)"
sidebar option, which had never actually worked.

**Fix**: `keyword_index.add(chunks)` now runs whenever chunks exist, independent
of `embedding_provider`; `vector_index.add(chunks)` remains gated on the
embedding block (it needs vectors attached first). Two regression tests were
added (`tests/test_ingest_hardening.py::TestKeywordIndexIndependentOfEmbedding`)
to lock in that BM25-only ingestion populates the keyword index and that
combined BM25+vector ingestion still populates both.

This is a useful illustration of why an end-to-end retrieval eval — not just
unit tests — is valuable: 1,183 unit tests were passing at the time, but none
of them exercised "ingest without an embedding provider, then actually search
the resulting BM25 index," which is exactly the configuration this bug lived
in.

---

## Ingest performance optimizations

Building and re-running this evaluation repeatedly (355-document corpus,
four chunk-size configs, a hybrid-retrieval config), plus a direct audit of
upload-size, memory, and context-window handling, surfaced four ingest-side
findings — three throughput bottlenecks and one memory-scaling defect. All
are now fixed. The benchmarks below use different methodologies, noted
explicitly wherever a live measurement wasn't possible or wasn't the right
tool for the claim being made.

### 1. Vision-LLM page transcription — sequential → parallel (simulated benchmark)

Pages escalated to the Vision-LLM tier (`PdfVisionParser`'s full-document
fallback, and `PdfHybridParser`'s per-page escalation for complex PDFs — see
the ingest-hardening work earlier in this project) were transcribed one HTTP
round-trip at a time. Each page's Vision call is independent and
network-bound, not CPU-bound, so this is embarrassingly parallel. Both
parsers now transcribe pages concurrently via a shared `ThreadPoolExecutor`
primitive (`PdfVisionParser.parse_pages_parallel`, default 8 workers,
configurable via `vision.max_workers` in `parser_candidates.yaml`).

**Methodology caveat**: this project's OpenAI account is subject to a shared
200k TPM rate limit, which made a live back-to-back sequential-vs-parallel
benchmark impractical (a full sequential run would consume rate-limit budget
needed for other testing). Instead, one real API call's latency was measured
(**2.06s/page**, `gpt-4o-mini`, full-page image transcription) and used to
simulate 16 pages under both worker counts:

| Workers | Time (16 pages, simulated at measured 2.06s/page latency) |
|---:|---:|
| 1 (sequential) | 33.0s |
| 8 (parallel) | 4.1s |
| **Speedup** | **8.0x** |

This is the expected ceiling for this workload shape (independent,
latency-dominated HTTP calls, no shared CPU/GPU resource to contend for) —
real-world speedup should track close to this since the bottleneck is network
wait, not local compute, but has not been verified against live back-to-back
API calls due to the rate-limit constraint above.

### 2. Batch ingest — eliminating O(N²) BM25 index rebuild (real benchmark)

`BM25LocalIndex.add()` retokenized and rebuilt the entire `rank_bm25.BM25Okapi`
model from scratch on every call, and `IngestPipeline` called `add()` once per
document. Ingesting a batch of *N* documents therefore did *O(N²)* work: the
1st document's chunks get tokenized once, but by the *N*th document, every
prior document's chunks are retokenized again on top of the new one.

**Fix**: `add()` gained a `rebuild: bool = True` parameter, and
`IngestPipeline.begin_batch()` / `end_batch()` context toggles a batch mode
that defers rebuilding (`add(chunks, rebuild=False)`) until the whole batch is
committed, then rebuilds **once**. `BatchIngestRunner` and the new
`ingest_many()` (see below) use this automatically.

This was benchmarked directly — no simulation — by re-ingesting the same
355-document, 6,477-chunk ASQA corpus used throughout this study, into two
fresh isolated indexes, with and without batch mode:

| Mode | Time (355 docs, 6,477 chunks) |
|---|---:|
| Before (`add()` per document, no batch mode) | 239.2s |
| After (`begin_batch()` / `end_batch()`) | 73.7s |
| **Speedup** | **3.2x** (69% time reduction) |

**Reading**: 3.2x is lower than the O(N²)→O(N) complexity change alone would
suggest, because index rebuild is only *part* of total ingest time — parsing,
cleaning, chunking, and DocStore writes are fixed per-document costs that
batch mode doesn't touch. The 165.5s saved is specifically the retokenization
work eliminated by rebuilding once instead of 355 times; this fraction grows
with corpus size, so the speedup would be larger on a bigger batch and smaller
on a tiny one.

### 3. File-level ingest concurrency

Alongside the O(N²) fix, `IngestPipeline._run()` was split into `_prepare()`
(parse/clean/chunk — read-only, side-effect-free, safe to run concurrently
across files) and `_commit()` (DocStore writes + index updates — must stay
serial, since SQLite has a single writer and the in-memory BM25/FAISS index
objects aren't thread-safe to mutate concurrently). The new
`IngestPipeline.ingest_many(paths, max_workers=N)` runs `_prepare()` for up to
*N* files at once in a thread pool, then commits sequentially in original
file order — overlapping CPU/IO-bound parsing (and any per-file Vision API
calls) across files instead of processing one file fully before starting the
next. Exposed via `rag/cli/ingest.py --workers` and used automatically by
`BatchIngestRunner` when `workers > 1`. Not separately benchmarked in this
study — the batch-mode number above (§2) isolates the index-rebuild fix
specifically; `_prepare()`/`_commit()` concurrency compounds with it on
multi-core machines but the combined effect was not measured here.

### 4. Bounding `ingest_many()` peak memory to `max_workers`

Auditing this system for memory and scaling risk (a natural follow-up
question once ingest was made concurrent) surfaced a real defect in the
concurrency design from §3: **peak memory during `ingest_many()` scaled with
the total batch size, not `max_workers`** — the concurrency knob controlled
parallelism but not memory, contrary to what its name implies.

**Root cause**: `ingest_many()` submitted *all* `N` files' `_prepare()` jobs
(parse → clean → chunk → embed) to the thread pool upfront, then drained
completions via `as_completed()`. `ThreadPoolExecutor` bounds how many jobs
run *concurrently* to `max_workers`, but places no limit on how many can
*finish* and sit fully computed inside their `Future` object waiting to be
consumed. For cheap/fast documents, worker threads race through the entire
queued backlog far faster than a single commit-writer thread can consume
results — so completed-but-uncommitted documents (each holding its full
parsed block list, chunk list, and any embedding vectors) can pile up to the
full batch size regardless of `max_workers`.

An initial fix attempt — moving the `_commit()` call inside the
`as_completed()` loop, so each result is committed as soon as it arrives —
looked plausible but is **not actually sufficient**, for the reason above:
completion can still race arbitrarily far ahead of consumption. A regression
test written specifically to measure this (tracking the number of
prepared-but-uncommitted documents at every point during a 12-file batch
with `max_workers=3`) caught it directly:

```python
assert peak_in_flight <= max_workers   # first fix attempt: FAILED, peak_in_flight == 12
```

**Real fix**: a sliding-window submission pattern. Instead of submitting all
`N` jobs upfront, only `max_workers` jobs are ever outstanding at once — the
next job is submitted only after a previous one's result has been consumed
and committed (`concurrent.futures.wait(..., return_when=FIRST_COMPLETED)`).
This provides genuine backpressure: submission is gated on consumption, so
memory is bounded by design, not by hoping the consumer keeps up.

```python
assert peak_in_flight <= max_workers   # sliding-window fix: PASSED
```

**Methodology note**: this is a *logical* bound proven by direct
instrumentation (a counter incremented on prepare-start and decremented on
commit-finish, asserted `<= max_workers` across the whole run) — not a
live memory-profiler measurement of actual process RSS in MB. The two are
related (bounding the count of in-flight fully-parsed documents bounds the
memory those documents occupy) but only the former was verified here.

**A related, independent gap in the same audit**: `LocalFileLoader.load()`
called `file_path.read_bytes()` unconditionally, with no size check —
reading an arbitrarily large file entirely into memory in one call, with no
streaming fallback. The Studio UI's 200MB upload cap is a Streamlit-layer
default that doesn't apply to the CLI/batch-ingest path at all, which had
*zero* protection. Fixed with a configurable `max_file_size_bytes` guard
(default 500MB) on `LocalFileLoader`, raising a clear `FileTooLargeError`
before any read is attempted, wired through `IngestPipeline`'s constructor
and reported as a normal `IngestResult(status="failed")` like any other
ingest error.

A third question from the same audit — whether the LLM-generation side has
an unbounded context-window risk — was checked and found **already correct**:
`LightContextPacker` enforces a `token_budget` (default 2048) with
deduplication and truncation before any context reaches the answer-generation
prompt. No fix was needed there.

---

## Known limitations of this study

- **Sample size**: 150 of 948 dev examples (503 of the resulting sub-questions).
  This is a pilot scoped for fast iteration, not a full-dataset benchmark.
  Numbers should be treated as indicative of *direction and magnitude*, not
  precise population estimates. Running the full dev split (948 examples) is
  a drop-in rerun with `--n-samples 948`.
- **Recall@k via substring match**, not ASQA's official answer-quality metric
  (string-overlap F1 against generated long-form answers). This study
  evaluates the *retrieval* stage only, not end-to-end generation quality.
- **Corpus scope**: only the Wikipedia pages ASQA's annotators actually cited
  for the sampled questions (356 URLs) were ingested — not full Wikipedia.
  This is deliberate (full Wikipedia embedding is infeasible on a laptop, ~21M
  passages in the academic ALCE benchmark), but it means the corpus is
  "easier" than a production web-scale retrieval setting: every question's
  answer is guaranteed to be somewhere in the ~6,000–25,000 indexed chunks,
  not lost in a much larger haystack.
- **Single embedding model tested**: only OpenAI `text-embedding-3-small` was
  evaluated for the hybrid configuration. The project's local `multilingual`
  embedding provider was not benchmarked here.
- **BM25-only chunk sweep, not hybrid chunk sweep**: the chunk-size sweep
  (128/256/512/768) used BM25-only; hybrid retrieval was only tested at the
  already-selected 512-token budget. It's possible the hybrid recall curve
  peaks at a different chunk size — not verified.
- **Vision parallelization speedup is simulated, not live-measured**: the
  8.0x number (§ [Ingest performance optimizations](#ingest-performance-optimizations))
  is derived from one real per-page API latency measurement replayed under
  simulated concurrency, not a live sequential-vs-parallel run — this
  project's shared 200k TPM OpenAI rate limit made the live comparison
  impractical. The BM25 batch-indexing speedup (3.2x), by contrast, is a
  direct, unsimulated benchmark on the real 355-document corpus.
- **`ingest_many()` memory bound is a logical proof, not a profiled
  measurement**: §4 verifies peak prepared-but-uncommitted document count
  stays `<= max_workers` via direct instrumentation in a test, not via a
  memory profiler (e.g. `tracemalloc`/`psutil`) reporting actual process RSS
  in MB on a realistic large batch. The two should track together, but only
  the former was measured.

---

## Reproducing these results

```bash
# Prerequisites
pip install datasets

# Chunk-size sweep (BM25-only, free, no API key needed)
python scripts/asqa_eval.py --n-samples 150 --token-budget 128 \
  --data-dir data/eval/asqa --output data/eval/asqa/pilot_tb128_results.json
python scripts/asqa_eval.py --n-samples 150 --token-budget 256 \
  --data-dir data/eval/asqa --output data/eval/asqa/pilot_tb256_results.json
python scripts/asqa_eval.py --n-samples 150 --token-budget 512 \
  --data-dir data/eval/asqa --output data/eval/asqa/pilot_tb512_results.json
python scripts/asqa_eval.py --n-samples 150 --token-budget 768 \
  --data-dir data/eval/asqa --output data/eval/asqa/pilot_tb768_results.json

# Hybrid retrieval at the selected chunk size (requires OPENAI_API_KEY, ~$0.10-0.20)
python scripts/asqa_eval.py --n-samples 150 --token-budget 512 --embedding openai \
  --data-dir data/eval/asqa --output data/eval/asqa/pilot_tb512_openai_results.json

# Batch-ingest speedup benchmark (§ Ingest performance optimizations, free,
# reuses the cached corpus from the runs above — no network or API key needed)
python scripts/bench_batch_ingest.py

# ingest_many() peak-memory bound (§4) and the file-size guard — both are
# regression tests, not standalone scripts:
pytest tests/test_ingest_hardening.py::TestIngestMany::test_commits_interleaved_not_batched_after_all_prepares -v
pytest tests/test_local_file_loader.py::TestLocalFileLoaderSizeGuard -v
```

The corpus HTML cache (`data/eval/asqa/corpus/`) is fetched once and reused
across all runs above — only the first invocation touches the network for
fetching. All output paths live under `data/eval/asqa/` and are excluded from
git (see `.gitignore`); production data in `data/rag.db` is never read or
written by this script.

Full per-question detail (question text, expected short answers, hit/miss at
each *k*) is saved in each run's output JSON under the `"detail"` key, for
error analysis.
