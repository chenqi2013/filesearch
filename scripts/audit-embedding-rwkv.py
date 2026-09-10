from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_REVISION = "13613c08d03c82b2fa4bd9b414e946132aa331b0"
EOS_ID = 65535


def upstream_source(repository, path):
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{UPSTREAM_REVISION}:{path}"],
        encoding="utf-8",
    )


def extract_definitions(source, names, namespace):
    parsed = ast.parse(source)
    selected = [
        node for node in parsed.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing upstream definitions: {names}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<upstream>", "exec"), namespace)


def load_reference(repository):
    namespace = {
        "torch": torch, "nn": nn, "F": F, "math": math,
        "pl": SimpleNamespace(LightningModule=nn.Module),
        "HEAD_SIZE": 64, "CHUNK_LEN": 16, "STOP_TOKEN_INDEX": 261,
    }
    extract_definitions(
        upstream_source(repository, "embedding/reranker/src/model.py"),
        ["RWKV7_OP"], namespace,
    )
    official_op = namespace["RWKV7_OP"]

    def cpu_operator(receptance, log_decay, key, value, context_key, context_value):
        state = torch.zeros((receptance.shape[0], 12, 64, 64))
        return official_op(
            receptance, log_decay, key, value, context_key, context_value, state
        )[0]

    namespace["RUN_CUDA_RWKV7g"] = cpu_operator
    extract_definitions(
        upstream_source(repository, "embedding/eval/src/model.py"),
        ["RWKV_Tmix_x070", "RWKV_CMix_x070", "Block", "RWKV",
         "NonlinearHead", "MultiEOSPooling", "MultiTaskHead"], namespace,
    )
    batch_namespace = dict(namespace, List=list, PromptType=SimpleNamespace(query="query"))
    extract_definitions(
        upstream_source(repository, "embedding/eval/custom_embedding_model.py"),
        ["VisualRWKVMTEBModel"], batch_namespace,
    )
    batch_namespace["EOS_INDEX"] = EOS_ID
    batch_namespace["PAD_INDEX"] = 0
    return namespace, batch_namespace["VisualRWKVMTEBModel"]


def load_exporter():
    path = Path(__file__).with_name("export-embedding-rwkv.py")
    spec = importlib.util.spec_from_file_location("rwkv_exporter", path)
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def candidate_operator(receptance, sigmoid_decay, key, value, context_key, context_value,
                       fix_decay=True, fix_order=True):
    batch_size, token_count, hidden_size = receptance.shape
    shaped = [tensor.reshape(batch_size, token_count, 12, 64)
              for tensor in (receptance, sigmoid_decay, key, value, context_key, context_value)]
    receptors, decays, keys, values, context_keys, context_values = shaped
    if fix_decay:
        decays = torch.exp(-math.exp(-0.5) * decays)
    state = torch.zeros((batch_size, 12, 64, 64))
    output = torch.empty_like(receptance)
    for token_index in range(token_count):
        decayed = state * decays[:, token_index, :, None, :]
        projected = (state if fix_order else decayed) @ context_keys[:, token_index, :, :, None]
        state = (decayed + projected @ context_values[:, token_index, :, None, :]
                 + values[:, token_index, :, :, None] @ keys[:, token_index, :, None, :])
        output[:, token_index] = (state @ receptors[:, token_index, :, :, None]).reshape(batch_size, hidden_size)
    return output


def forward_candidate(model, tokens, eos_mask=None):
    hidden = model.embedding(tokens)
    value_first = torch.zeros_like(hidden)
    for layer in model.layers:
        hidden, value_first = layer(hidden, value_first)
    hidden = F.layer_norm(hidden, [768], model.output_norm_weight, model.output_norm_bias)
    if eos_mask is None:
        pooled = hidden[:, -1]
    else:
        pooled = hidden.masked_fill(~eos_mask.unsqueeze(-1), 0).sum(1) / eos_mask.sum(1, keepdim=True)
    projected = F.linear(F.relu(F.linear(pooled, model.retr_fc1_weight)), model.retr_fc2_weight)
    return F.layer_norm(pooled + projected, [768], model.retr_norm_weight, model.retr_norm_bias)


def verify(namespace, state, candidate):
    torch.manual_seed(20260909)
    tensors = [torch.randn(2, 7, 768) * 0.1 for _ in range(6)]
    receptance, logits, key, value, context_key, context_value = tensors
    log_decay = -F.softplus(-logits) - 0.5
    initial = torch.zeros((2, 12, 64, 64))
    expected, _ = namespace["RWKV7_OP"](
        receptance, log_decay, key, value, context_key, context_value, initial
    )
    checks = {}
    for mode, fix_decay, fix_order in [("legacy", False, False), ("decay_only", True, False),
                                       ("order_only", False, True), ("corrected", True, True)]:
        actual = candidate_operator(receptance, logits.sigmoid(), key, value, context_key,
                                    context_value, fix_decay, fix_order)
        checks[mode] = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    decay_expected = torch.exp(-torch.exp(log_decay))
    decay_actual = torch.exp(-math.exp(-0.5) * logits.sigmoid())
    torch.testing.assert_close(decay_actual, decay_expected)
    checks["decay_identity_max_error"] = float((decay_actual - decay_expected).abs().max())
    args = SimpleNamespace(n_embd=768, n_layer=12, vocab_size=65536, dim_att=768,
                           head_size_a=64, head_size_divisor=8, dropout=0.0, grad_cp=0)
    reference = namespace["RWKV"](args).eval()
    missing = reference.load_state_dict(
        {key[5:]: value for key, value in state.items() if key.startswith("rwkv.")},
        strict=False,
    )
    if missing.missing_keys != ["head.weight"] or missing.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint mismatch: {missing}")
    head = namespace["MultiTaskHead"](768, 768).eval()
    head.load_state_dict({key[5:]: value for key, value in state.items() if key.startswith("head.")})
    tokens = torch.randint(1, 65000, (2, 32))
    tokens[:, 15] = EOS_ID
    tokens[:, 31] = EOS_ID
    with torch.inference_mode():
        corrected_output = candidate(tokens)
        torch.testing.assert_close(forward_candidate(candidate, tokens), corrected_output)
        reference_layers, candidate_layers = [], []
        reference_hooks = [block.register_forward_hook(
            lambda module, inputs, output: reference_layers.append(output[0].detach()))
            for block in reference.blocks]
        candidate_hooks = [layer.register_forward_hook(
            lambda module, inputs, output: candidate_layers.append(output[0].detach()))
            for layer in candidate.layers]
        hidden = reference(reference.emb(tokens))
        mask = tokens == EOS_ID
        pooled = hidden.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)
        expected_embedding = head(pooled, torch.full((2,), 2))
        actual_embedding = forward_candidate(candidate, tokens, mask)
        for hook in reference_hooks + candidate_hooks:
            hook.remove()
        checks["per_layer_max_errors"] = []
        for expected_layer, actual_layer in zip(reference_layers, candidate_layers):
            torch.testing.assert_close(actual_layer, expected_layer, atol=1e-4, rtol=3e-4)
            checks["per_layer_max_errors"].append(float((actual_layer - expected_layer).abs().max()))
        if len(checks["per_layer_max_errors"]) != 12:
            raise AssertionError("Expected all 12 layers to be compared")
        checks["full_model_max_error"] = float((actual_embedding - expected_embedding).abs().max())
        checks["full_model_min_cosine"] = float(F.cosine_similarity(actual_embedding, expected_embedding).min())
        torch.testing.assert_close(actual_embedding, expected_embedding, atol=3e-5, rtol=3e-4)
        partial_tokens = tokens[:, :21].clone()
        partial_tokens[:, -1] = EOS_ID
        partial_mask = partial_tokens == EOS_ID
        hidden = reference(reference.emb(partial_tokens))
        pooled = hidden.masked_fill(~partial_mask.unsqueeze(-1), 0).sum(1) / partial_mask.sum(1, keepdim=True)
        expected_embedding = head(pooled, torch.full((2,), 2))
        padded_tokens = F.pad(partial_tokens, (11, 0), value=261)
        padded_mask = F.pad(partial_mask, (11, 0), value=False)
        actual_embedding = forward_candidate(candidate, padded_tokens, padded_mask)
        torch.testing.assert_close(actual_embedding, expected_embedding, atol=3e-5, rtol=3e-4)
        checks["stop_padding_max_error"] = float((actual_embedding - expected_embedding).abs().max())
    print(json.dumps({"verification": checks}), flush=True)
    return checks


def metrics(scores, documents, queries, qrels):
    ndcg, recall, hit_rate, accuracy = [], [], [], []
    for row, query in zip(scores, queries):
        relevant = qrels[query["_id"]]
        ranking = torch.argsort(row, descending=True, stable=True)[:10].tolist()
        hits = [documents[index]["_id"] in relevant for index in ranking]
        dcg = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
        ndcg.append(dcg / ideal)
        recall.append(sum(hits) / len(relevant))
        hit_rate.append(float(any(hits)))
        accuracy.append(float(hits[0]))
    return {name: sum(values) / len(values) for name, values in
            [("ndcg@10", ndcg), ("recall@10", recall), ("hit_rate@10", hit_rate), ("accuracy@1", accuracy)]}


def encode(texts, candidate, tokenizer, batch_builder, mode, batch_size):
    tokenized = [tokenizer.encode(text, add_eos=False) for text in texts]
    order = sorted(range(len(texts)), key=lambda index: len(tokenized[index]))
    embeddings = torch.empty((len(texts), 768))
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            indexes = order[offset:offset + batch_size]
            if mode == "official_input":
                batch = batch_builder._build_batch([texts[index] for index in indexes])
                tokens = batch["query_ids"]
                mask = tokens == EOS_ID
                valid_mask = torch.zeros_like(mask)
                valid_mask[mask] = batch["query_eos_mask"].flatten()
                stop_padding = (-tokens.shape[1]) % 16
                tokens = F.pad(tokens, (stop_padding, 0), value=261)
                valid_mask = F.pad(valid_mask, (stop_padding, 0), value=False)
            else:
                batches = [tokenized[index][:1023] + [EOS_ID] for index in indexes]
                max_length = max(map(len, batches))
                tokens = torch.tensor([[0] * (max_length - len(ids)) + ids for ids in batches])
                valid_mask = None
            vectors = forward_candidate(candidate, tokens, valid_mask)
            embeddings[indexes] = F.normalize(vectors, dim=1)
            if offset == 0 or (offset // batch_size) % 25 == 0:
                print(f"{mode}: {min(offset + batch_size, len(order))}/{len(order)} "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-documents", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=["legacy", "decay_only", "order_only", "corrected", "official_input"],
                        default=["legacy", "decay_only", "corrected", "official_input"])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    namespace, batch_class = load_reference(args.repository)
    exporter = load_exporter()
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    candidate = exporter.EmbeddingRwkvTiny(state).eval()
    with args.checkpoint.open("rb") as checkpoint_file:
        checkpoint_hash = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result = {"upstream_revision": UPSTREAM_REVISION, "device": "cpu", "dtype": "float32",
              "checkpoint": str(args.checkpoint.resolve()), "batch_size": args.batch_size,
              "checkpoint_sha256": checkpoint_hash,
              "protocol": {"single_eos_max_tokens": 1024, "official_ctx_len": 1024,
                           "official_eos_chunk_size": 512, "length_sorted_batches": True,
                           "cuda_executed": False, "official_mteb_runner_executed": False},
              "threads": args.threads, "verification": verify(namespace, state, candidate),
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset:
        import pyarrow.parquet as parquet

        sys.path.insert(0, str(args.repository / "package" / "src"))
        from rwkv_emb.tokenizer import RWKVTokenizer

        tokenizer = RWKVTokenizer()
        batch_builder = batch_class.__new__(batch_class)
        batch_builder.cfg = SimpleNamespace(ctx_len=1024, eos_chunk_size=512)
        batch_builder.device = torch.device("cpu")
        batch_builder.tokenizer = SimpleNamespace(encode=lambda text: tokenizer.encode(text, add_eos=False))
        rows = {kind: parquet.read_table(args.dataset / kind / "NanoSciFact-00000-of-00001.parquet").to_pylist()
                for kind in ["corpus", "queries", "qrels"]}
        documents, queries = rows["corpus"], rows["queries"]
        qrels = {}
        for row in rows["qrels"]:
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
        if args.limit_documents:
            relevant_ids = set().union(*qrels.values())
            documents = ([doc for doc in documents if doc["_id"] in relevant_ids]
                         + [doc for doc in documents if doc["_id"] not in relevant_ids]
                         [:max(0, args.limit_documents - len(relevant_ids))])
        result["dataset"] = {"name": "NanoSciFact", "documents": len(documents), "queries": len(queries),
                             "limited": bool(args.limit_documents), "qrels": len(rows["qrels"]),
                             "corpus_sha256": hashlib.sha256(json.dumps(documents, sort_keys=True).encode()).hexdigest()}
        for mode in args.modes:
            exporter.Rwkv7Function.apply = lambda *inputs: candidate_operator(
                *inputs, fix_decay=mode != "legacy" and mode != "order_only",
                fix_order=mode not in ["legacy", "decay_only"])
            started = time.perf_counter()
            document_embeddings = encode([doc["text"] for doc in documents], candidate, tokenizer,
                                         batch_builder, mode, args.batch_size)
            query_texts = [query["text"] for query in queries]
            if mode == "official_input":
                query_texts = [f"Instruct: Given a query, retrieve documents that answer the query\nQuery: {text}"
                               for text in query_texts]
            query_embeddings = encode(query_texts, candidate, tokenizer, batch_builder, mode, args.batch_size)
            entry = {"mode": mode, "embedding_seconds": time.perf_counter() - started,
                     **metrics(query_embeddings @ document_embeddings.T, documents, queries, qrels)}
            result["results"].append(entry)
            print(json.dumps(entry), flush=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
