from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import struct
import time
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.codex-tmp/embeddingrwkv-official-20261028'
REVISION = '3c306736c58550f4be6d384be068512ba9bfbd72'
DEFAULT_OUTPUT = ROOT / '.codex-tmp/official-rwkv-20261028'
CHECKPOINT = ROOT / '.codex-tmp/EmbeddingRWKV/models/rwkv0b1-emb-curriculum.mirror.pth'
WEIGHTS = ROOT / '.codex-tmp/rwkv-emb-c-upstream/models/embedding-fp32.bin'
DATABASE = ROOT / '.codex-tmp/semantic-generalization-20261027/run1/data/search.db'
APPLICATION = ROOT / '.codex-tmp/semantic-generalization-20261027/run6/result.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class OfficialCPU:
    def __init__(self, output, threads, serial=False):
        self.dll_directory = os.add_dll_directory(
            'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin')
        name = 'rwkv_emb_serial.dll' if serial else 'rwkv_emb.dll'
        self.library = ct.CDLL(str(output / name))
        self.integers = ct.POINTER(ct.c_int32)
        self.bytes = ct.POINTER(ct.c_uint8)
        self.floats = ct.POINTER(ct.c_float)
        handle = ct.c_void_p
        signatures = {
            'rwkv_threads': ([ct.c_int], None),
            'rwkv_load': ([ct.c_char_p], handle),
            'rwkv_free': ([handle], None),
            'rwkv_load_tokenizer': ([handle, ct.c_char_p], ct.c_int),
            'rwkv_tokenize': ([handle, ct.c_char_p, self.integers, ct.c_int], ct.c_int),
            'rwkv_encode_tokens': ([handle, self.integers, self.bytes, ct.c_int,
                                    ct.c_int, ct.c_int, self.floats, self.floats], ct.c_int),
            'rwkv_encode_texts': ([handle, ct.POINTER(ct.c_char_p), ct.c_int,
                                   ct.c_int, ct.c_int, ct.c_int, self.floats], ct.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result
        self.library.rwkv_threads(threads)
        self.handle = self.library.rwkv_load(str(WEIGHTS).encode('ascii'))
        if not self.handle:
            raise RuntimeError('Official C weight loading failed')
        tokenizer = WEIGHTS.with_name('tokenizer.bin')
        if self.library.rwkv_load_tokenizer(self.handle, str(tokenizer).encode('ascii')):
            raise RuntimeError('Official C tokenizer loading failed')

    def close(self):
        self.library.rwkv_free(self.handle)

    def tokenize(self, text):
        encoded = text.encode('utf-8')
        if b'\0' in encoded:
            raise ValueError('C tokenizer cannot accept embedded NUL')
        values = np.empty(len(encoded) + 1, np.int32)
        count = self.library.rwkv_tokenize(self.handle, encoded,
                                         values.ctypes.data_as(self.integers), len(values))
        if count < 0:
            raise RuntimeError('Official tokenization failed')
        return values[:count].tolist()

    def tokens(self, ids, mask=None, trace=False):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        mask = np.ascontiguousarray(ids == 65535 if mask is None else mask, dtype=np.uint8)
        batch, length = ids.shape
        values = np.empty((batch, 768), np.float32)
        traces = np.empty((14, batch, length, 768), np.float32) if trace else None
        status = self.library.rwkv_encode_tokens(
            self.handle, ids.ctypes.data_as(self.integers), mask.ctypes.data_as(self.bytes),
            batch, length, 2, values.ctypes.data_as(self.floats),
            traces.ctypes.data_as(self.floats) if trace else None)
        if status:
            raise RuntimeError(f'Official C inference failed: {status}')
        return (values, traces) if trace else values

    def evaluate_protocol(self, texts):
        values = np.empty((len(texts), 768), np.float32)
        strings = (ct.c_char_p * len(texts))(*(text.encode('utf-8') for text in texts))
        status = self.library.rwkv_encode_texts(
            self.handle, strings, len(texts), 2048, 512, 2, values.ctypes.data_as(self.floats))
        if status:
            raise RuntimeError(f'Official evaluation preprocessing failed: {status}')
        return values


def inputs():
    application = json.loads(APPLICATION.read_text(encoding='utf-8'))
    with sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True) as database:
        documents = dict(database.execute('SELECT id,name FROM documents ORDER BY id'))
        rows = database.execute(
            'SELECT id,document_id,chunk_id,text,embedding FROM semantic_passages ORDER BY id').fetchall()
    passages = [dict(id=identity, document_id=document, name=documents[document],
                     chunk_id=chunk, text=text) for identity, document, chunk, text, vector in rows]
    persisted = np.asarray([np.frombuffer(row[4], dtype='<f4') for row in rows])
    cases = application['cases']
    texts = [case['query'] for case in cases] + [row['name'][:160] + '\n' + row['text'] for row in passages]
    return application, passages, persisted, texts


def identity(output, texts):
    return dict(upstream_revision=REVISION, source_sha256=digest(UPSTREAM / 'rwkv-emb.c/rwkv_emb.c'),
                dll_sha256=digest(output / 'rwkv_emb.dll'), checkpoint_sha256=digest(CHECKPOINT),
                official_huggingface_checkpoint_sha256='9033eec92f163d1a710474977fa3fb68b7ee04697e0e961d83434743bd256a15',
                weights_sha256=digest(WEIGHTS), database_sha256=digest(DATABASE),
                application_result_sha256=digest(APPLICATION),
                input_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
                context=1024, eos=65535, pooling='single EOS', head='RETR', precision='FP32')


def rust_vectors(texts, model, threads, batch_size=4):
    benchmark = module('rwkv_benchmark', ROOT / 'scripts/benchmark-long-txt.py')
    arguments = SimpleNamespace(root=ROOT, rwkv_model=model,
                                runtime=ROOT / 'src-tauri/binaries/onnxruntime.dll', threads=threads,
                                spinning=False, profile=None, inter_threads=1, batch_size=batch_size,
                                batch_token_budget=2048)
    return benchmark.rwkv_run(arguments, texts)


def validate(output, runtime, threads):
    import torch
    from torch.nn import functional as functional

    torch.set_num_threads(threads)
    audit = module('rwkv_official_audit', ROOT / 'scripts/audit-embedding-rwkv.py')
    audit.UPSTREAM_REVISION = REVISION
    namespace, _ = audit.load_reference(UPSTREAM)
    state = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=True)
    tensor_checks = 0
    with WEIGHTS.open('rb') as stream:
        header = struct.unpack('<8I', stream.read(32))
        for _ in range(header[6]):
            name = stream.read(128).split(b'\0')[0].decode('ascii')
            dimensions, *shape_and_count = struct.unpack('<5IQ', stream.read(28))
            shape, count = shape_and_count[:dimensions], shape_and_count[-1]
            actual = np.frombuffer(stream.read(count * 4), dtype='<f4').reshape(shape)
            expected = state[name].float()
            if expected.ndim == 2 and name.endswith('.weight') and name != 'rwkv.emb.weight':
                expected = expected.T
            np.testing.assert_array_equal(actual, expected.numpy())
            tensor_checks += 1
        assert stream.read(1) == b''
    config = SimpleNamespace(n_embd=768, n_layer=12, vocab_size=65536, dim_att=768,
                             head_size_a=64, head_size_divisor=8, dropout=0.0, grad_cp=0)
    reference = namespace['RWKV'](config).eval()
    loaded = reference.load_state_dict(
        {key[5:]: value for key, value in state.items() if key.startswith('rwkv.')}, strict=False)
    assert loaded.missing_keys == ['head.weight'] and not loaded.unexpected_keys
    head = namespace['MultiTaskHead'](768, 768).eval()
    head.load_state_dict({key[5:]: value for key, value in state.items() if key.startswith('head.')})
    checks = []
    generator = np.random.default_rng(20261028)
    for length in [16, 64, 256]:
        ids = generator.integers(1, 60000, (2, length), dtype=np.int32)
        ids[:, length // 2 - 1] = 65535
        ids[:, -1] = 65535
        mask = ids == 65535
        started = time.perf_counter()
        actual, traces = runtime.tokens(ids, trace=True)
        c_seconds = time.perf_counter() - started
        intermediate = []
        hooks = [block.register_forward_hook(
            lambda block, arguments, result: intermediate.append(result[0].detach().numpy()))
                 for block in reference.blocks]
        with torch.inference_mode():
            token_tensor = torch.from_numpy(ids.astype(np.int64))
            embedded = reference.emb(token_tensor)
            hidden = reference(embedded)
            selected = torch.from_numpy(mask)
            pooled = hidden.masked_fill(~selected.unsqueeze(-1), 0).sum(1) / selected.sum(1, keepdim=True)
            expected = functional.normalize(head(pooled, torch.full((2,), 2)), dim=-1).numpy()
        for hook in hooks:
            hook.remove()
        layer_errors = []
        for actual_layer, expected_layer in zip(traces, [embedded.numpy(), *intermediate, hidden.numpy()]):
            np.testing.assert_allclose(actual_layer, expected_layer, atol=0.003, rtol=0.003)
            layer_errors.append(float(np.max(np.abs(actual_layer - expected_layer))))
        np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0.001)
        np.testing.assert_array_equal(runtime.tokens(ids), actual)
        checks.append(dict(tokens=length, c_seconds=c_seconds, layer_errors=layer_errors,
                           embedding_max_abs=float(np.max(np.abs(actual - expected))),
                           cosine=(actual * expected).sum(1).tolist()))
        print(json.dumps(checks[-1]), flush=True)
    application, passages, _, texts = inputs()
    benchmark = module('rwkv_benchmark_validation', ROOT / 'scripts/benchmark-long-txt.py')
    tokenizer = benchmark.RwkvTokenizer(ROOT / 'assets/models/embedding-rwkv-tiny/rwkv_vocab.bin')
    selected_texts = ['', 'Hello world!', '赔款', '以前很火的老歌', 'naïve café 🧠\nline two',
                      '中文内容' * 1000, 'English text. ' * 2000]
    selected_texts += texts[:12]
    document_indices = defaultdict(list)
    for index, passage in enumerate(passages):
        document_indices[passage['document_id']].append(index)
    for indices in document_indices.values():
        for index in sorted({indices[0], indices[len(indices) // 2], indices[-1]}):
            selected_texts.append(texts[len(application['cases']) + index])
    c_values = []
    for text in selected_texts:
        sequence = runtime.tokenize(text)[:1023] + [65535]
        assert sequence == tokenizer.encode(text)
        c_values.append(runtime.tokens([sequence])[0])
    c_values = np.asarray(c_values)
    timing, values = rust_vectors(selected_texts, ROOT / '.codex-tmp/rwkv-corrected-export/model.onnx', threads)
    cosine = (c_values * values).sum(1)
    np.testing.assert_allclose(c_values, values, atol=0.0001, rtol=0.003)
    validation = dict(tensors_equal_to_checkpoint=tensor_checks, official_equation_checks=checks,
                      tokenizer_and_fp32_onnx_samples=len(selected_texts),
                      rust_fp32_min_cosine=float(cosine.min()),
                      rust_fp32_max_abs=float(np.abs(c_values - values).max()), rust_timing=timing)
    save_json(output / 'validation.json', validation)
    print(json.dumps(validation, ensure_ascii=False), flush=True)


def encode(output, runtime, texts, batch_size):
    manifest = identity(output, texts)
    assert manifest['checkpoint_sha256'] == manifest['official_huggingface_checkpoint_sha256']
    cache = output / 'official-vectors.npz'
    if cache.exists():
        saved = np.load(cache)
        assert json.loads(str(saved['identity'])) == manifest
        values, completed = saved['vectors'], saved['completed']
    else:
        values = np.zeros((len(texts), 768), np.float32)
        completed = np.zeros(len(texts), bool)
    started = time.perf_counter()
    grouped = defaultdict(list)
    for index, text in enumerate(texts):
        if not completed[index]:
            sequence = runtime.tokenize(text)[:1023] + [65535]
            grouped[len(sequence)].append((index, sequence))
    batches = []
    for length, group in sorted(grouped.items()):
        size = min(batch_size, max(1, 2048 // length))
        batches.extend(group[start:start + size] for start in range(0, len(group), size))
    last_report = time.perf_counter()
    for batch in batches:
        result = runtime.tokens([sequence for index, sequence in batch])
        for (index, sequence), value in zip(batch, result):
            values[index] = value
            completed[index] = True
        if time.perf_counter() - last_report > 20:
            np.savez(cache, identity=json.dumps(manifest), vectors=values, completed=completed)
            print(f'Official C: {completed.sum()}/{len(texts)} texts; elapsed {time.perf_counter()-started:.1f}s', flush=True)
            last_report = time.perf_counter()
    np.savez(cache, identity=json.dumps(manifest), vectors=values, completed=completed)
    save_json(output / 'identity.json', manifest)
    print(f'Official C finished: {completed.sum()} texts in this session {time.perf_counter()-started:.1f}s', flush=True)


def protocol_diagnostic(output, runtime):
    application, passages, persisted, texts = inputs()
    wanted = {'送过来的东西坏了谁掏钱', '货不对板能不能退回去', '交货少了应该怎么补',
              '出了纠纷去哪个法院', '断网以后还能找电脑里的资料吗', '自动识别文件类型并提取元数据的工具',
              '榜单', '赔款', '钱款', '不想继续合作要提前多久说', '身体活力的分数是怎么算的'}
    query_vectors = np.load(output / 'int8-singleton-queries.npz')['vectors']
    raw = rank_scores(application['cases'], passages, query_vectors @ persisted.T)
    cases = [case for case in application['cases'] if case['query'] in wanted]
    selected = set()
    for row in raw:
        if row['query'] in wanted:
            selected.update(value['index'] for value in row['top_passages'][:5])
            if row['evidence_best_index'] is not None:
                selected.add(row['evidence_best_index'])
    indices = sorted(selected)
    scope_path = output / 'protocol-result.json'
    if scope_path.exists():
        frozen_ids = set(json.loads(scope_path.read_text(encoding='utf-8'))['scope']['passage_ids'])
        indices = [index for index, passage in enumerate(passages) if passage['id'] in frozen_ids]
    candidates = [passages[index] for index in indices]
    cache = output / 'protocol-vectors.npz'
    cache_identity = dict(cases=cases, passage_ids=[row['id'] for row in candidates],
                          selection='union of current INT8 top5 passages and best frozen-quote passage per query; not full corpus',
                          source_sha256=digest(UPSTREAM / 'rwkv-emb.c/rwkv_emb.c'), weights_sha256=digest(WEIGHTS))
    variant_texts = {
        'quickstart_with_filename': [case['query'] for case in cases]
                                   + [row['name'][:160] + '\n' + row['text'] for row in candidates],
        'quickstart_body_only': [case['query'] for case in cases] + [row['text'] for row in candidates],
        'official_evaluation_multi_eos': [case['query'] for case in cases]
                                       + [row['name'][:160] + '\n' + row['text'] for row in candidates],
    }
    saved = dict(identity=np.array(json.dumps(cache_identity, ensure_ascii=False)))
    if cache.exists():
        previous = np.load(cache)
        assert json.loads(str(previous['identity'])) == cache_identity
        saved.update({key: previous[key] for key in previous.files if key != 'identity'})
    report = dict(scope=cache_identity, results={})
    for variant, values in variant_texts.items():
        if variant not in saved:
            vectors = []
            started = time.perf_counter()
            for index, text in enumerate(values):
                value = (runtime.evaluate_protocol([text])[0] if variant == 'official_evaluation_multi_eos'
                         else runtime.tokens([runtime.tokenize(text)[:1023] + [65535]])[0])
                vectors.append(value)
                if (index + 1) % 10 == 0:
                    print(f'{variant}: {index+1}/{len(values)}; {time.perf_counter()-started:.1f}s', flush=True)
            saved[variant] = np.asarray(vectors)
            np.savez(cache, **saved)
        values = saved[variant]
        count = len(cases)
        rows = rank_scores(cases, candidates, values[:count] @ values[count:].T)
        report['results'][variant] = [dict(query=row['query'], document_rank=row['document_rank'],
                                           evidence_rank=row['evidence_rank'], evidence_score=row['evidence_score'],
                                           top1=row['top_passages'][0]) for row in rows]
    save_json(output / 'protocol-result.json', report)
    for query_index, case in enumerate(cases):
        print(json.dumps(dict(query=case['query'], variants={name: dict(document_rank=rows[query_index]['document_rank'],
                                                                       evidence_rank=rows[query_index]['evidence_rank'],
                                                                       evidence_score=rows[query_index]['evidence_score'])
                                                           for name, rows in report['results'].items()}), ensure_ascii=False), flush=True)


def inspect_tokens(output, runtime):
    application, passages, persisted, texts = inputs()
    lengths = np.asarray([len(runtime.tokenize(text)) + 1 for text in texts])
    count = len(application['cases'])
    report = dict(query_max_tokens=int(lengths[:count].max()),
                  passage_max_tokens=int(lengths[count:].max()),
                  queries_truncated=int((lengths[:count] > 1024).sum()),
                  passages_truncated=int((lengths[count:] > 1024).sum()),
                  length_percentiles=np.percentile(lengths[count:], [0, 5, 50, 95, 100]).tolist(),
                  lengths=lengths.tolist())
    save_json(output / 'tokenization.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'lengths'}), flush=True)


def oracle_diagnostic(output, runtime):
    protocol = json.loads((output / 'protocol-result.json').read_text(encoding='utf-8'))
    application, passages, persisted, texts = inputs()
    query_vectors = np.load(output / 'int8-singleton-queries.npz')['vectors']
    raw = {row['query']: row for row in rank_scores(application['cases'], passages, query_vectors @ persisted.T)}
    report = dict(scope='Oracle diagnostic using known evidence location. Not a retrieval accuracy benchmark or a production algorithm.', rows=[])
    for case in protocol['scope']['cases']:
        frozen_path = output / 'oracle-result.json'
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
            frozen_row = next(row for row in frozen['rows'] if row['query'] == case['query'])
            index = next(index for index, passage in enumerate(passages)
                         if passage['name'] == frozen_row['name']
                         and passage['text'] == frozen_row['variants']['full_body']['text'])
        else:
            index = raw[case['query']]['evidence_best_index']
        text = passages[index]['text']
        offsets = [position for position, character in enumerate(text) if not character.isspace()]
        match = normalized(text).index(normalized(case['evidence']))
        middle = offsets[match + len(normalized(case['evidence'])) // 2]
        values = {'full_body': text}
        for length in [200, 96]:
            start = max(0, min(middle - length // 2, len(text) - length))
            values[f'evidence_centered_{length}'] = text[start:start + length]
        query = runtime.tokens([runtime.tokenize(case['query']) + [65535]])[0]
        variants = {}
        for name, value in values.items():
            embedded = runtime.tokens([runtime.tokenize(value)[:1023] + [65535]])[0]
            variants[name] = dict(score=float(query @ embedded), text=value)
        row = dict(query=case['query'], evidence=case['evidence'], name=passages[index]['name'], variants=variants)
        report['rows'].append(row)
        print(json.dumps(dict(query=case['query'], scores={name: value['score'] for name, value in variants.items()}), ensure_ascii=False), flush=True)
    save_json(output / 'oracle-result.json', report)


def validate_persisted(output, runtime, threads):
    application, passages, persisted, texts = inputs()
    selected = set(np.linspace(0, len(passages) - 1, 32, dtype=int).tolist())
    indices_by_document = defaultdict(list)
    for index, passage in enumerate(passages):
        indices_by_document[passage['document_id']].append(index)
    for indices in indices_by_document.values():
        selected.update([indices[0], indices[len(indices) // 2], indices[-1]])
    indices = sorted(selected)
    count = len(application['cases'])
    cache = output / 'persisted-samples.npz'
    if cache.exists():
        previous = np.load(cache)
        np.testing.assert_array_equal(previous['indices'], indices)
        actual = previous['vectors']
        timing = json.loads(str(previous['timing']))
    else:
        timing, actual = rust_vectors([texts[count + index] for index in indices],
                                      ROOT / 'assets/models/embedding-rwkv-tiny/model.onnx', threads)
        np.savez(cache, indices=indices, vectors=actual, timing=json.dumps(timing))
    expected = persisted[indices]
    cosine = np.sum(actual * expected, axis=1)
    print(json.dumps(dict(persisted_min_cosine=float(cosine.min()),
                          lowest=[dict(index=indices[index], name=passages[indices[index]]['name'], cosine=float(cosine[index]))
                                  for index in np.argsort(cosine)[:10]]), ensure_ascii=False), flush=True)
    serial = OfficialCPU(output, 1, serial=True)
    errors = []
    try:
        for index in indices[:6] + indices[-6:]:
            ids = [runtime.tokenize(texts[count + index])[:1023] + [65535]]
            parallel_value = runtime.tokens(ids)
            serial_value = serial.tokens(ids)
            np.testing.assert_allclose(parallel_value, serial_value, atol=0.0001, rtol=0.003)
            errors.append(float(np.abs(parallel_value - serial_value).max()))
    finally:
        serial.close()
    report = dict(passage_indices=indices, samples=len(indices),
                  persisted_vs_fresh_int8_min_cosine=float(cosine.min()),
                  persisted_vs_fresh_int8_max_abs=float(np.abs(actual - expected).max()),
                  serial_vs_parallel_c_max_abs=max(errors), serial_vs_parallel_samples=len(errors),
                  timing=timing)
    save_json(output / 'persisted-validation.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def encode_int8(output, threads):
    application, passages, persisted, texts = inputs()
    count = len(application['cases'])
    model = ROOT / 'assets/models/embedding-rwkv-tiny/model.onnx'
    query_path = output / 'int8-singleton-queries.npz'
    if query_path.exists():
        assert str(np.load(query_path)['model_sha256']) == digest(model)
    else:
        query_timing, queries = rust_vectors(texts[:count], model, threads, batch_size=1)
        np.savez(query_path, vectors=queries, model_sha256=digest(model))
        save_json(output / 'int8-singleton-query-timing.json', query_timing)
    passage_path = output / 'int8-fresh-passages.npz'
    if passage_path.exists():
        previous = np.load(passage_path)
        assert str(previous['model_sha256']) == digest(model)
        assert str(previous['input_sha256']) == hashlib.sha256(json.dumps(texts[count:], ensure_ascii=False).encode()).hexdigest()
        return
    timing, passages = rust_vectors(texts[count:], model, threads)
    np.savez(passage_path, vectors=passages, model_sha256=digest(model),
             input_sha256=hashlib.sha256(json.dumps(texts[count:], ensure_ascii=False).encode()).hexdigest())
    save_json(output / 'int8-fresh-passage-timing.json', timing)


def export_core_comparison(output):
    application, passages, persisted, texts = inputs()
    cache = np.load(output / 'official-vectors.npz')
    assert cache['completed'].all()
    count = len(application['cases'])
    official = cache['vectors']
    queries = np.load(output / 'int8-singleton-queries.npz')['vectors']
    official_rows = rank_scores(application['cases'], passages, official[:count] @ official[count:].T)
    stored_rows = rank_scores(application['cases'], passages, queries @ persisted.T)
    report = dict(official=metrics(official_rows), stored=metrics(stored_rows), rows=[])
    lookup = {row['query']: row for row in application['searches'] if row['mode'] == 'semantic'}
    for fp32, stored in zip(official_rows, stored_rows):
        if fp32['targets']:
            report['rows'].append(dict(query=fp32['query'], kind=fp32['kind'],
                                       official_doc=fp32['document_rank'], stored_doc=stored['document_rank'],
                                       official_evidence=fp32['evidence_rank'], stored_evidence=stored['evidence_rank'],
                                       application_doc=lookup[fp32['query']]['target_rank']))
    save_json(output / 'core-comparison.json', report)
    print(json.dumps(dict(official=report['official'], stored=report['stored']), ensure_ascii=False), flush=True)
    for row in report['rows']:
        if row['application_doc'] is None or row['official_doc'] != row['stored_doc']:
            print(json.dumps(row, ensure_ascii=False), flush=True)


def normalized(text):
    return ''.join(text.lower().split())


def metrics(rows):
    result = {}
    for kind in ['new_positive', 'controls']:
        selected = [row for row in rows if row['kind'] == kind]
        result[kind] = dict(
            total=len(selected),
            document_top1=sum(row['document_rank'] == 1 for row in selected),
            document_hit3=sum(row['document_rank'] is not None and row['document_rank'] <= 3 for row in selected),
            document_hit10=sum(row['document_rank'] is not None and row['document_rank'] <= 10 for row in selected),
            evidence_label_available=sum(row['evidence_passage_count'] > 0 for row in selected),
            evidence_passage_top1=sum(row['evidence_rank'] == 1 for row in selected),
            evidence_passage_hit10=sum(row['evidence_rank'] is not None and row['evidence_rank'] <= 10 for row in selected),
            evidence_passage_hit100=sum(row['evidence_rank'] is not None and row['evidence_rank'] <= 100 for row in selected))
    return result


def rank_scores(cases, passages, scores):
    indices_by_document = defaultdict(list)
    for index, passage in enumerate(passages):
        indices_by_document[passage['document_id']].append(index)
    ranked = []
    for case, row in zip(cases, scores):
        order = np.argsort(-row, kind='stable')
        inverse = np.empty(len(order), np.int64)
        inverse[order] = np.arange(1, len(order) + 1)
        documents = []
        for indices in indices_by_document.values():
            best = max(indices, key=lambda index: row[index])
            documents.append(dict(name=passages[best]['name'], score=float(row[best]),
                                  passage_index=best, text=passages[best]['text']))
        documents.sort(key=lambda item: -item['score'])
        targets = case['targets']
        target_rank = next((rank for rank, document in enumerate(documents, 1)
                            if document['name'] in targets), None)
        evidence = normalized(case.get('evidence', ''))
        relevant = [index for index, passage in enumerate(passages)
                    if passage['name'] in targets and evidence and evidence in normalized(passage['text'])]
        best_evidence = min(relevant, key=lambda index: inverse[index]) if relevant else None
        evidence_rank = int(inverse[best_evidence]) if best_evidence is not None else None
        ranked.append(dict(**case, document_rank=target_rank, evidence_rank=evidence_rank,
                           evidence_passage_count=len(relevant),
                           evidence_best_index=best_evidence,
                           evidence_score=float(row[best_evidence]) if best_evidence is not None else None,
                           top_documents=documents,
                           top_passages=[dict(index=int(index), rank=rank, score=float(row[index]),
                                              name=passages[index]['name'], text=passages[index]['text'])
                                         for rank, index in enumerate(order[:10], 1)]))
    return ranked


def analyze(output, threads):
    application, passages, persisted, texts = inputs()
    cases = application['cases']
    count = len(cases)
    official_cache = np.load(output / 'official-vectors.npz')
    assert official_cache['completed'].all(), 'Full official corpus encoding is not complete'
    assert json.loads(str(official_cache['identity'])) == identity(output, texts)
    official = official_cache['vectors']
    query_cache = output / 'int8-singleton-queries.npz'
    production_model = ROOT / 'assets/models/embedding-rwkv-tiny/model.onnx'
    if query_cache.exists():
        cached = np.load(query_cache)
        assert str(cached['model_sha256']) == digest(production_model)
        queries = cached['vectors']
    else:
        timing, queries = rust_vectors(texts[:count], production_model, threads, batch_size=1)
        np.savez(query_cache, vectors=queries, model_sha256=digest(production_model))
        save_json(output / 'int8-singleton-query-timing.json', timing)
    official_rows = rank_scores(cases, passages, official[:count] @ official[count:].T)
    int8_rows = rank_scores(cases, passages, queries @ persisted.T)
    fresh_cache = np.load(output / 'int8-fresh-passages.npz')
    assert str(fresh_cache['model_sha256']) == digest(production_model)
    assert str(fresh_cache['input_sha256']) == hashlib.sha256(json.dumps(texts[count:], ensure_ascii=False).encode()).hexdigest()
    fresh = fresh_cache['vectors']
    fresh_rows = rank_scores(cases, passages, queries @ fresh.T)
    passage_cosine = np.sum(official[count:] * persisted, axis=1)
    query_cosine = np.sum(official[:count] * queries, axis=1)
    describe = lambda values: {name: float(np.percentile(values, percentile))
                              for name, percentile in [('min', 0), ('p05', 5), ('median', 50), ('p95', 95), ('max', 100)]}
    application_lookup = {row['query']: row for row in application['searches'] if row['mode'] == 'semantic'}
    application_rows = [application_lookup[case['query']] for case in cases]
    assert [row['query'] for row in application_rows] == texts[:count]
    contrasts = []
    for case, fp32, int8, current, app in zip(cases, official_rows, int8_rows, fresh_rows, application_rows):
        if case['targets']:
            contrasts.append(dict(query=case['query'], kind=case['kind'], targets=case['targets'],
                                  official_document_rank=fp32['document_rank'],
                                  int8_document_rank=int8['document_rank'],
                                  fresh_int8_document_rank=current['document_rank'],
                                  application_document_rank=app['target_rank'],
                                  official_evidence_rank=fp32['evidence_rank'],
                                  int8_evidence_rank=int8['evidence_rank'],
                                  fresh_int8_evidence_rank=current['evidence_rank'],
                                  official_evidence_score=fp32['evidence_score'],
                                  int8_evidence_score=int8['evidence_score'],
                                  official_top1=fp32['top_documents'][0]['name'],
                                  int8_top1=int8['top_documents'][0]['name']))
    result = dict(
        scope=dict(documents=len({row['document_id'] for row in passages}), passages=len(passages),
                   novel_passages=sum(row['name'] == '小说——超级龙宠.txt' for row in passages), queries=count,
                   document_ranking='max passage cosine; all documents retained without threshold',
                   evidence_metric='rank of a target-document passage containing the frozen evidence quote; incomplete relevance labels',
                   application='historical frozen run6, same corpus; includes anchors, query confirmation and filtering'),
        identity=json.loads(str(official_cache['identity'])),
        int8_model_sha256=digest(production_model),
        int8_encoder_sha256=digest(ROOT / 'target/release/examples/rwkv-embed.exe'),
        int8_query_batch_size=1,
        int8_fresh_passage_batch_size=4,
        vector_cosine_fp32_vs_int8=dict(queries=describe(query_cosine), passages=describe(passage_cosine)),
        vector_cosine_fp32_vs_fresh_int8=dict(passages=describe(np.sum(official[count:] * fresh, axis=1))),
        vector_cosine_stored_vs_fresh_int8=dict(passages=describe(np.sum(persisted * fresh, axis=1))),
        summary=dict(official_fp32_raw=metrics(official_rows), production_int8_raw=metrics(int8_rows),
                     fresh_int8_raw=metrics(fresh_rows),
                     application=application['summary']['semantic']), contrasts=contrasts,
        official_fp32=official_rows, production_int8=int8_rows, fresh_int8=fresh_rows)
    save_json(output / 'result.json', result)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2), flush=True)
    for row in contrasts:
        if row['application_document_rank'] is None or row['official_document_rank'] != row['int8_document_rank']:
            print(json.dumps(row, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Diagnostic only: official CPU FP32 vs production RWKV')
    parser.add_argument('--phase', choices=['validate', 'encode', 'encode-int8', 'analyze', 'protocol', 'inspect', 'oracle', 'persisted', 'core-report'], required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'analyze':
        analyze(args.output, args.threads)
        return
    if args.phase == 'core-report':
        export_core_comparison(args.output)
        return
    if args.phase == 'encode-int8':
        encode_int8(args.output, args.threads)
        return
    runtime = OfficialCPU(args.output, args.threads, serial=args.phase == 'validate')
    try:
        if args.phase == 'validate':
            validate(args.output, runtime, args.threads)
        elif args.phase == 'protocol':
            protocol_diagnostic(args.output, runtime)
        elif args.phase == 'inspect':
            inspect_tokens(args.output, runtime)
        elif args.phase == 'oracle':
            oracle_diagnostic(args.output, runtime)
        elif args.phase == 'persisted':
            validate_persisted(args.output, runtime, args.threads)
        else:
            encode(args.output, runtime, inputs()[3], args.batch_size)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
