# Corrected RWKV integration and benchmark

## Implementation scope

- `crates/search-core/src/rwkv.rs`: corrected CPU ONNX custom operator, with scalar
  and AVX2 implementations. Existing ONNX sigmoid inputs are converted once to
  `exp(-exp(-0.5) * sigmoid(z))`. The rank-one update uses the pre-decay state.
- `crates/search-core/examples/rwkv-embed.rs`: isolated JSON-lines embedding process.
  This is an explicit Cargo example, not a replacement for the desktop sidecar.
- `scripts/export-embedding-rwkv.py`: corrected Python recurrence; verification now
  uses pinned upstream formulas rather than the modified local CPU fallback.
- `scripts/benchmark-rwkv-nanobeir.py`: complete-corpus CPU retrieval evaluation,
  checkpoint/data/binary hashes, timing, sampled process peak RSS, per-dataset
  checkpoints, vector caches, and configuration-checked `--resume`.
- `scripts/verify-rwkv-retrieval-metrics.py`: independently recomputes metrics from
  cached vectors using the original sentence-transformers metric methods.

The desktop remains on Qwen. No index or bundled weight was replaced, no installer
was built, and no model file is added to Git. The future corrected RWKV profile is
`rwkv7-tiny-corrected-single-eos-v2`; deployment must rebuild previous RWKV vectors.

## Verification

- Rust decay identity, pre-decay state recurrence, scalar/AVX2 equivalence and
  upstream golden-value tests: 4 passed.
- Existing main binary tests: 20 passed, 5 explicitly ignored.
- End-to-end Rust + existing ONNX vs corrected Python exporter: normalized-vector
  maximum absolute error `8.05e-7`, minimum cosine `1.0` on the validation batch.
- The exporter itself is compared against the pinned upstream evaluation model
  running the upstream PyTorch recurrence: maximum error `3.46e-6`.
- All 12 intermediate block outputs also pass FP32 tolerance checks; last-layer
  absolute error is `1.07e-4` before output LayerNorm, with the final embedding
  error returning to `3.46e-6`.
- Upstream source revision: `13613c08d03c82b2fa4bd9b414e946132aa331b0`.
- CUDA hardware execution has not been tested on this CPU-only machine.
- Actual re-export to `.codex-tmp/rwkv-corrected-export/` succeeded. Both ONNX and
  vocabulary are byte-identical to the bundled artifacts, confirming the sigmoid
  input contract is unchanged. ONNX SHA-256:
  `9d0985b53784997f07359a465d4e35da9d307265081bb69ac1eaf11c9b677b83`.

## Evaluation protocol

- Dataset: `sentence-transformers/NanoBEIR-en`, all 13 English subsets.
- Dataset revision: `beb106fbcfaa599c508c667041bf8c85fd78736b`.
- All documents and all labeled queries, not the earlier 300-document subset.
- Existing Tiny ONNX checkpoint, corrected Rust WKV, CPU FP32.
- Batch size 4, length sorting, left-zero padding, maximum 1024 tokens including EOS.
- Single final EOS pooling, `[RETR]` head, normalized cosine retrieval.
- No query instruction, no reranker, no multi-EOS repetition.
- Binary positive judgments, nDCG@10, Recall@10, HitRate@10, Accuracy@1.
- Macro-average over all 13 datasets, not a document-weighted average.
- Runtime measurements include tokenization and inference separately. Process peak
  RSS is sampled every 100 ms and cumulative across the run; it is not GPU VRAM.
- The additional 20-document/40-query Chinese set is a small synthetic diagnostic,
  not a substitute for real local-document relevance judgments.

This evaluates the corrected application's simple embedding protocol on NanoBEIR,
not the upstream MTEB runner's complete input pipeline. It must not be claimed as
an exact reproduction of the paper's 59.10 result.

## Completed results

Hardware: Intel Core i7-13700H. Only CPU execution is measured.

All 13 datasets completed: 56,723 documents and 649 queries (Touche2020 has 49).
Macro nDCG@10 is **48.9155%**. The summed corpus/query inference time is 7,005.2
seconds (116.75 minutes); tokenization, model loading, downloads, validation and
metrics are excluded. Other verification commands briefly shared the machine,
so this run is not an isolated throughput benchmark. Peak sampled Rust RSS across
the run is 1,274.7 MiB.

| Dataset | nDCG@10 | Recall@10 | Accuracy@1 |
| --- | ---: | ---: | ---: |
| SciFact | 70.66% | 84.00% | 56.00% |
| ArguAna | 54.63% | 88.00% | 18.00% |
| ClimateFEVER | 27.33% | 32.00% | 26.00% |
| DBPedia | 41.04% | 25.49% | 42.00% |
| FEVER | 69.54% | 85.67% | 50.00% |
| FiQA2018 | 43.32% | 52.19% | 36.00% |
| HotpotQA | 58.82% | 66.00% | 58.00% |
| MSMARCO | 34.94% | 52.00% | 18.00% |
| NFCorpus | 22.62% | 9.82% | 24.00% |
| NQ | 44.10% | 65.00% | 28.00% |
| QuoraRetrieval | 92.73% | 99.00% | 88.00% |
| SCIDOCS | 36.56% | 38.37% | 42.00% |
| Touche2020 | 39.61% | 24.55% | 42.86% |

All 13 datasets' metrics were recomputed using the original sentence-transformers
metric methods: nDCG@10, Accuracy@1 and HitRate@10 match exactly; the maximum
Recall@10 difference is 1.11e-16 (floating-point averaging).

Exact run data and independent metric checks are preserved alongside this report
in `rwkv-corrected-nanobeir-results-20260909.json` and
`rwkv-corrected-nanobeir-metrics-20260909.json`. Large vectors and model artifacts
remain under ignored local directories.

The remaining difference from the paper's 59.10 cannot be assigned to a specific
cause by this run. Single-EOS pooling, no instruction, 1024-token truncation,
batch padding, dtype, dataset revision and the paper's actual evaluation settings
still need controlled comparison. A full multi-EOS/instruction ablation is a
separate experiment, not something this simple-protocol benchmark establishes.

Full NanoSciFact (2,919 documents, 50 queries): nDCG@10 70.66%, Recall@10 84%,
Accuracy@1 56%. Corpus inference took 650.5 seconds, query inference 0.93 seconds,
and sampled Rust process peak RSS was 1,144.6 MiB. These are CPU batch benchmark
times, not per-file desktop indexing times or a controlled comparison with older
runtime timings.

The sentence-transformers v3.4.1 metric methods (commit
`7d52a069e0b37d976b3ed3f674a6180436c27574`) reproduce all four reported SciFact
metrics exactly from the cached vectors. The old approximately 3% nDCG result
used a broken implementation and cannot be treated as the Tiny model's capability.

The synthetic Chinese diagnostic (20 documents, 40 queries) achieves 100% nDCG@10
and Accuracy@1. This small, easy set is useful only as a regression check, not an
estimate of accuracy on arbitrary real documents.

Full ArguAna (3,635 documents, 50 queries): nDCG@10 54.63%, Recall@10 88%,
Accuracy@1 18%. Its metrics also match the original sentence-transformers methods.

## Commands

From the project root:

```powershell
cargo test -p search-core --example rwkv-embed
cargo test -p search-core --bin search-core
cargo build --release -p search-core --example rwkv-embed
python -B -u scripts/benchmark-rwkv-nanobeir.py --repository .codex-tmp/EmbeddingRWKV --checkpoint .codex-tmp/EmbeddingRWKV/models/rwkv0b1-emb-curriculum.mirror.pth --model assets/models/embedding-rwkv-tiny/model.onnx --executable target/release/examples/rwkv-embed.exe --runtime .codex-tmp/onnxruntime-1.22.0/runtimes/win-x64/native/onnxruntime.dll --data .codex-tmp/NanoBEIR-data --output .codex-tmp/rwkv-nanobeir-corrected-full.json --local-eval .codex-tmp/semantic-retrieval-eval.json
python -B scripts/verify-rwkv-retrieval-metrics.py --results .codex-tmp/rwkv-nanobeir-corrected-full.json --data .codex-tmp/NanoBEIR-data --output .codex-tmp/rwkv-metrics-verification.json
```

Python dependencies: torch, numpy, pyarrow, psutil and requests. Local upstream Git
checkout and model files are prerequisites. The benchmark downloads only pinned
dataset parquet files, not model weights. Each Rust request has a 180-second
response timeout, and the child process is terminated on benchmark failure.

Add `--datasets SciFact` for a focused complete NanoSciFact run; add `--resume` to
reuse finished datasets only when model, executable, data and protocol match.
