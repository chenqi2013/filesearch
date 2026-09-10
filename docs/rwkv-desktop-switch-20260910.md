# RWKV desktop switch (2026-09-10)

## Runtime

- EmbeddingRWKV Tiny, RWKV7, 768-dimensional normalized embeddings.
- Reuses the corrected WKV kernel in `crates/search-core/src/rwkv.rs`.
- Official World vocabulary; no query instruction; one final EOS (65535).
- Maximum input: 1024 tokens including EOS.
- CPU ONNX session: four intra-op threads, with existing AVX2 WKV acceleration.
- CUDA remains optional for other ONNX nodes; WKV still runs on CPU.
- Profile: `rwkv7-tiny-corrected-unpadded-document-chunks-v3`.
- Changing profile invalidates old document and chunk vectors and automatically
  reindexes configured directories. Source files are not modified.

## Padding check

The ONNX model has no padding mask. Left-padding shorter inputs with token zero
changes recurrent state. A mixed-length batch produced cosine 0.86264020 against
single-input inference for the query "系统如何防止网络攻击".

The desktop adapter now batches only equal token lengths, with at most four
inputs and 2048 total tokens per batch. This avoids padding without changing the
model artifact or benchmark runner. Tests cover original output order, sequence
limits, EOS, singleton consistency, and equal-length multi-input inference.
The five mixed-length test texts now match singleton outputs to cosine > 0.9999.
Batching throughput can differ from older padded benchmarks.

## Verification

- Standard core tests: 26 passed, three explicit slow tests skipped.
- Two model-backed tests run separately: both passed.
- Local retrieval fixture: 20 documents, 40 queries, newly generated vectors.
- Top-1: 39/40 (97.5%); Recall@3: 40/40 (100%); MRR@3: 0.9875.
- Remaining rank-2 query: "候选人在正式上班前要经过哪些环节？".
  Expected: recruitment; top result: training.
- CPU smoke test: model load approximately 998 ms; five short text embeddings
  approximately 104 ms. These are not large-document indexing timings.

These retrieval results measure raw cosine ranking, not the desktop's combined
semantic/lexical scoring. They use unpadded inputs; the earlier 100% local test
used mixed-length padded batches. Neither small-fixture result establishes
accuracy on arbitrary user files. NanoBEIR was not rerun during this switch.
