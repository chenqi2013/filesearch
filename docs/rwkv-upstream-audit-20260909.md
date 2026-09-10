# EmbeddingRWKV upstream audit (2026-09-09)

## Scope and conclusion

This document records the initial audit. The subsequent Rust correction and full
benchmark run are documented in `rwkv-corrected-benchmark-20260909.md`. References
to the broken exporter below describe its pre-fix version.

The old RWKV integration has two confirmed WKV math errors. Previous comparisons
against the locally modified Python CPU fallback were not independent: that
fallback contains the same errors. They cannot establish equivalence with upstream
CUDA, and the old retrieval scores must not be used to judge the model's quality.

This audit adds an isolated, reproducible diagnostic script. It does not replace
the current Qwen implementation, rewrite existing indexes, overwrite ONNX weights,
or build an installer. Historical Rust code below refers to commit `3e3b59e`;
the working tree's `embedding.rs` currently implements Qwen.

## Provenance

- Upstream: https://github.com/howard-hou/EmbeddingRWKV
- Pinned commit: `13613c08d03c82b2fa4bd9b414e946132aa331b0`.
- `git ls-remote` confirmed this was upstream HEAD on 2026-09-09.
- Checkpoint: `rwkv0b1-emb-curriculum.pth` (local suffix `.mirror.pth`).
- Size: 476,851,895 bytes; 12 layers, 768 dimensions, 12 heads of size 64.
- SHA-256: `9033eec92f163d1a710474977fa3fb68b7ee04697e0e961d83434743bd256a15`.
- The hash matches the LFS object reported by the official Hugging Face API:
  https://huggingface.co/api/models/howard-hou/EmbeddingRWKV/tree/main
- Runtime for this audit: PyTorch 2.6.0, CPU FP32, 2 threads, batch size 8.
- CUDA is unavailable on this machine. No native CUDA execution was performed.

The audit script reads model definitions directly with `git show <commit>:<path>`.
It does not import the modified `package/src/rwkv_emb/reference/rwkv7.py` fallback.
It extracts the upstream evaluation model and the upstream reranker's PyTorch
`RWKV7_OP`, substituting only the CUDA operation with that upstream CPU formula.
This verifies mathematical agreement, not bitwise FP16/BF16 CUDA equivalence.

## Step-by-step differences

### 1. Decay transformation: confirmed error

Upstream package Python emits `u = sigmoid(z)`. Its CUDA kernel then computes:

```text
d = exp(-exp(-0.5) * u)
```

The evaluation/training implementation uses an equivalent representation:

```text
w = -softplus(-z) - 0.5
d = exp(-exp(w))
```

These are equivalent because `exp(-softplus(-z)) = sigmoid(z)`.
This is not an accidental double transformation in upstream code.

The local exporter, historical Rust operator, and modified Python fallback use
`sigmoid(z)` directly as the multiplicative decay. Inspection of the bundled ONNX
confirmed that all 12 custom WKV nodes receive their second input directly from
a `Sigmoid` node. There is no hidden exponentiation in that graph.

Source anchors at the pinned upstream revision:

- `package/src/rwkv_emb/reference/rwkv7.py`: `RWKV_x070_TMix_seq_batch`.
- `package/src/rwkv_emb/reference/cuda/rwkv7_state_fwd_fp16.cu`: lines 36, 43-55.
- `embedding/eval/src/model.py`: `RWKV_Tmix_x070.forward`.
- `embedding/eval/cuda/wkv7_cuda.cu`: lines 21, 28-39.
- Local `scripts/export-embedding-rwkv.py`: `RwkvLayer.time_mix`.

### 2. State update order: confirmed error

Using row-major value-by-key state matrices, upstream computes:

```text
projection = S_previous @ a
S_next = S_previous * d + projection @ b + v @ k
y = S_next @ r
```

The old implementation instead computes:

```text
S_decayed = S_previous * d
projection = S_decayed @ a
S_next = S_decayed + projection @ b + v @ k
```

The projection must use the pre-decay state. This difference exists in the Python
exporter, the local CPU fallback, and both historical Rust scalar and AVX2 paths.
Scalar-versus-AVX2 agreement alone therefore cannot establish correctness.

The upstream reranker's `embedding/reranker/src/model.py::RWKV7_OP` independently
implements the same pre-decay formula as both CUDA kernels.

### 3. Tokenization and retrieval head

Both pipelines use the RWKV vocabulary and EOS ID 65535. The local export selects
the correct `[RETR]` nonlinear head: residual MLP followed by LayerNorm. The final
similarity calculation uses normalized vectors and cosine/dot product.

The corrected candidate, using the exporter's existing layer weights and head,
matches the independently loaded upstream evaluation model end to end. The audit
allows only the unused language-model `head.weight` to be absent from the checkpoint;
other missing/unexpected RWKV parameters cause failure.

### 4. Pooling and input construction: protocol differences

- Package quick start: one EOS; pool EOS hidden states, then apply the retrieval head.
- Current exporter: last token only; equivalent to one-EOS pooling when EOS is last.
- Upstream MTEB wrapper: multiple EOS positions, pooled before the retrieval head.
- At `ctx_len=1024`, `eos_chunk_size=512`, short texts repeat twice with two EOS tokens.
- Longer texts are split into multiple EOS segments, not simply truncated to 1024.
- `_build_batch` caps raw tokens at `ctx_len * 8` and masks artificial padding EOS.
- Evaluation `RWKV.forward` left-pads with STOP ID 261 to a multiple of 16 before
  its CUDA kernel, then removes those outputs. That prefix still affects state.
- Query instructions are optional in the runner; the checked-in shell example
  enables them. The default retrieval instruction is
  `Instruct: Given a query, retrieve documents that answer the query\nQuery: {query}`.
- Corpus encoding concatenates title and text when available. The local NanoSciFact
  parquet contains only `_id` and `text`, so no separate title is being omitted.
- The runner defaults to `ctx_len=1024`; `scripts/run_mteb.sh` specifies 2048.
  There is no basis to call every 1024-token experiment an exact paper reproduction.

The diagnostic `official_input` mode uses the actual upstream `_build_batch`,
default 1024/512 configuration, masked multi-EOS pooling, STOP padding, and query
instruction. It retains length-sorted CPU batches for tractability. It is not the
official MTEB runner or a full replication of the paper's evaluation settings.

### 5. Metrics and paper comparison

Earlier local scripts labeled `any(top_10_hits)` as Recall@10. That is HitRate@10.
The audit computes recall as `relevant_retrieved / total_relevant` per query and
reports both. It also computes nDCG@10 and Accuracy@1 separately.

NanoSciFact alone is not the full multi-dataset NanoBEIR benchmark. Paper Table 4
lists `EmbeddingRWKV 0.1B = 59.10` separately from the reranker rows. Although the
table is in the reranking section and reranker evaluation uses Top-100 candidates,
that does not establish that the 59.10 embedding baseline itself includes reranking.
Neither treating 59.10 as this single dataset's expected score nor attributing it
to reranking is justified without the corresponding complete run configuration.

## Numerical verification

Synthetic WKV comparison against the unmodified upstream PyTorch formula:

| Candidate | Maximum absolute error |
| --- | ---: |
| Old decay and old state order | 0.0210180 |
| Correct decay only | 0.00272910 |
| Correct state order only | 0.0218991 |
| Both corrected | 7.45e-9 |

Additional assertions:

- Decay algebra identity: max error 5.96e-8.
- Full 12-layer model, multi-EOS pooling and retrieval head: max error 3.46e-6;
  minimum cosine 1.0 on the two deterministic test sequences.
- Non-aligned sequence with official STOP padding: max error 4.69e-6.
- At audit time the legacy candidate was checked against the then-current exporter's
  forward output. After correction, that assertion checks the corrected exporter.

## Retrieval ablation

Diagnostic subset: all 50 NanoSciFact queries, 300 documents selected to retain
all their relevant documents plus deterministic distractors. This is an easier
diagnostic corpus, not a benchmark score comparable with the original 2,919-document
run or with published NanoBEIR. All modes use the same subset and labels.

| Mode | nDCG@10 | Recall@10 | Accuracy@1 | Embedding seconds |
| --- | ---: | ---: | ---: | ---: |
| Legacy math, single EOS | 3.54% | 10% | 0% | 158.0 |
| Correct decay only, single EOS | 89.51% | 96% | 82% | 177.1 |
| Both math corrections, single EOS | 90.12% | 96% | 84% | 160.8 |
| Both corrections, upstream 1024/512 input + instruction | 90.68% | 96% | 84% | 322.8 |

Timing covers corpus and query embedding in this Python CPU diagnostic, not desktop
indexing throughput. The decisive result is the controlled improvement from fixing
decay; multi-EOS or weight tuning is not required to explain the previous low score.
The combined input changes add only 0.56 percentage points of nDCG on this subset
while approximately doubling time. This does not isolate pooling from instruction,
STOP padding or long-text processing, and is not evidence of a universal trade-off.

## Reproduction

From the project root, with PyTorch and pyarrow installed:

```powershell
python scripts/audit-embedding-rwkv.py --repository .codex-tmp/EmbeddingRWKV --checkpoint .codex-tmp/EmbeddingRWKV/models/rwkv0b1-emb-curriculum.mirror.pth --output .codex-tmp/rwkv-audit-verification.json
python -u scripts/audit-embedding-rwkv.py --repository .codex-tmp/EmbeddingRWKV --checkpoint .codex-tmp/EmbeddingRWKV/models/rwkv0b1-emb-curriculum.mirror.pth --dataset .codex-tmp/NanoBEIR-data --limit-documents 300 --output .codex-tmp/rwkv-audit-ablation.json
```

Omit `--limit-documents` for the complete local NanoSciFact corpus. Use
`--modes corrected` for a focused corrected-formula run. The JSON records the
checkpoint hash, commit, runtime settings, checks, corpus identity and metrics.

## Safe integration order

1. Preserve the current Qwen working tree; use an isolated RWKV integration path.
2. Define the custom operator's input contract explicitly. Existing ONNX files pass
   sigmoid decay, so exponentiation belongs in the operator unless the graph and
   operator are versioned and regenerated together. Do not apply it twice.
3. Fix the state order in Python, Rust scalar, Rust AVX2 and any future CUDA kernel.
4. Add golden tests against upstream formulas, not only against another local port.
5. Run the full 2,919-document NanoSciFact set, then multi-dataset NanoBEIR and the
   local Chinese document test set before making model-selection claims.
6. Evaluate multi-EOS quality/latency separately; do not assume it always improves
   local search enough to justify additional tokens.
7. If deploying corrected RWKV, bump the embedding profile and rebuild all affected
   document/chunk vectors. Old and corrected vectors are not interchangeable.
