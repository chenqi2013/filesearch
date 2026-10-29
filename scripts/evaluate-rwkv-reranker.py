from __future__ import annotations

import argparse
import ast
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.codex-tmp/embeddingrwkv-official-20261028'
DEFAULT_OUTPUT = ROOT / '.codex-tmp/rwkv-reranker-20261028'
DATABASE = ROOT / '.codex-tmp/semantic-generalization-20261027/run1/data/search.db'
TEMPLATE = 'Instruct: Given a query, retrieve documents that answer the query\nDocument: {document}\nQuery: {query}'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def build(output):
    source = (UPSTREAM / 'rwkv-emb.c/rwkv_emb.c').read_text(encoding='utf-8')
    original = source
    source = source.replace('const float *v,const float *kk,const float *a,int B,int T,int C) {',
                            'const float *v,const float *kk,const float *a,int B,int T,int C,float *states) {')
    source = source.replace('memcpy(out+off,y,sizeof(y));\n        }',
                            'memcpy(out+off,y,sizeof(y));\n        }\n'
                            '        if(states) for(int row=0;row<64;row++) for(int col=0;col<64;col++)\n'
                            '            states[(((size_t)bb*H+h)*64+row)*64+col]=state[col*64+row];')
    source = source.replace('int B,int T,int task,float *out,float *trace) {',
                            'int B,int T,int task,float *out,float *trace,float *states) {')
    source = source.replace('rwkv_recurrence(y,r,w,k,v,kk,a,B,T,C);',
                            'rwkv_recurrence(y,r,w,k,v,kk,a,B,T,C,states?states+(size_t)li*B*(C/64)*64*64:NULL);')
    source = source.replace('task,out,NULL);', 'task,out,NULL,NULL);')
    assert source != original
    assert source.count('states[(((size_t)bb*H+h)*64+row)*64+col]=state[col*64+row];') == 1
    assert source.count('states?states+(size_t)li*B*(C/64)*64*64:NULL') == 1
    path = output / 'rwkv_state.c'
    path.write_text(source, encoding='utf-8')
    compiler = Path('C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang.exe')
    setup = Path('C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat')
    definitions = output / 'libiomp5md.def'
    definitions.write_text('LIBRARY libiomp5md.dll\nEXPORTS\n__kmpc_global_thread_num\n__kmpc_fork_call\n__kmpc_for_static_init_4\n__kmpc_for_static_init_4u\n__kmpc_for_static_init_8\n__kmpc_for_static_init_8u\n__kmpc_for_static_fini\nomp_set_num_threads\nomp_set_dynamic\n', encoding='ascii')
    command = (f'call "{setup}" >nul\n'
               f'lib /nologo /machine:x64 /def:"{definitions}" /out:"{output / "libiomp5md.lib"}"\n'
               f'"{compiler}" -O3 -march=native -fopenmp=libiomp5 -ffp-contract=fast -shared -DRWKV_NO_MAIN '
               f'"{path}" -L"{output}" -o "{output / "rwkv_state_intel.dll"}"\n')
    batch = output / 'build-state.cmd'
    batch.write_text(command, encoding='utf-8')
    result = subprocess.run(['cmd', '/c', str(batch)], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    save(output / 'state-build.json', dict(upstream_revision='3c306736c58550f4be6d384be068512ba9bfbd72',
         original_sha256=hashlib.sha256(original.encode()).hexdigest(),
         adapted_sha256=hashlib.sha256(source.encode()).hexdigest(),
         modifications='Expose final per-layer WKV states in PyTorch value/key layout; embedding math unchanged'))


def load_head(path):
    source = UPSTREAM / 'package/src/rwkv_emb/model.py'
    wanted = {'RWKV7_OP', 'RWKV_Tmix_x070', 'RWKV_CMix_x070', 'Block', 'ReRanker'}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    namespace = dict(torch=torch, nn=nn, F=functional, HEAD_SIZE=64, math=__import__('math'))
    exec(compile(tree, str(source), 'exec'), namespace)
    weights = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    state = {name.removeprefix('reranker.'): value.float() for name, value in weights.items() if name.startswith('reranker.')}
    if 'token.weight' in state:
        state['emb.weight'] = state.pop('token.weight')
    layers = max(int(name.split('.')[1]) for name in state if name.startswith('blocks.')) + 1
    arguments = SimpleNamespace(n_embd=768, n_layer=layers, head_size_a=64, dim_att=768, head_size_divisor=8)
    with torch.device('meta'):
        head = namespace['ReRanker'](arguments)
    head.load_state_dict(state, strict=True, assign=True)
    return head.eval(), weights.get('reranker_layer_idx', list(range(12)))


class StateEncoder:
    def __init__(self, output, threads):
        spec = importlib.util.spec_from_file_location('official_comparison', ROOT / 'scripts/compare-official-rwkv.py')
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        self.encoder = comparison.OfficialCPU.__new__(comparison.OfficialCPU)
        self.encoder.dll_directory = os.add_dll_directory('C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin')
        self.encoder.library = ct.CDLL(str(output / 'rwkv_state_intel.dll'))
        self.encoder.integers = ct.POINTER(ct.c_int32)
        self.encoder.bytes = ct.POINTER(ct.c_uint8)
        self.encoder.floats = ct.POINTER(ct.c_float)
        signatures = {
            'rwkv_threads': ([ct.c_int], None),
            'rwkv_load': ([ct.c_char_p], ct.c_void_p),
            'rwkv_free': ([ct.c_void_p], None),
            'rwkv_load_tokenizer': ([ct.c_void_p, ct.c_char_p], ct.c_int),
            'rwkv_tokenize': ([ct.c_void_p, ct.c_char_p, self.encoder.integers, ct.c_int], ct.c_int),
            'rwkv_encode_tokens': ([ct.c_void_p, self.encoder.integers, self.encoder.bytes, ct.c_int, ct.c_int,
                                   ct.c_int, self.encoder.floats, self.encoder.floats, self.encoder.floats], ct.c_int),
        }
        for name, (arguments, result) in signatures.items():
            method = getattr(self.encoder.library, name)
            method.argtypes, method.restype = arguments, result
        self.encoder.library.rwkv_threads(threads)
        self.encoder.handle = self.encoder.library.rwkv_load(str(comparison.WEIGHTS).encode('ascii'))
        assert self.encoder.handle
        assert self.encoder.library.rwkv_load_tokenizer(self.encoder.handle, str(comparison.WEIGHTS.with_name('tokenizer.bin')).encode('ascii')) == 0

    def encode(self, texts):
        sequences = [self.encoder.tokenize(text)[:1023] + [65535] for text in texts]
        length = max(map(len, sequences))
        ids = np.array([[0] * (length - len(sequence)) + sequence for sequence in sequences], dtype=np.int32)
        mask = np.ascontiguousarray(ids == 65535, dtype=np.uint8)
        states = np.empty((12, len(texts), 12, 64, 64), np.float32)
        vectors = np.empty((len(texts), 768), np.float32)
        status = self.encoder.library.rwkv_encode_tokens(
            self.encoder.handle, ids.ctypes.data_as(self.encoder.integers), mask.ctypes.data_as(self.encoder.bytes),
            len(texts), length, 2, vectors.ctypes.data_as(self.encoder.floats), None, states.ctypes.data_as(self.encoder.floats))
        assert status == 0, status
        return vectors, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--windows', type=int, default=16)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--queries', nargs='*')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'rwkv_state_intel.dll').exists():
        build(output)
    weights = args.weights or DEFAULT_OUTPUT / 'rwkv0b1-reranker-idx1610.pth'
    torch.set_num_threads(args.threads)
    head, layers = load_head(weights)
    encoder = StateEncoder(output, args.threads)
    with sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True) as database:
        names = dict(database.execute('SELECT id,name FROM documents'))
        rows = database.execute('SELECT document_id,text,embedding FROM semantic_passages ORDER BY id').fetchall()
    vectors = np.array([np.frombuffer(row[2], dtype='<f4') for row in rows])
    application = json.loads((ROOT / '.codex-tmp/search-evidence-20261028/final/result.json').read_text(encoding='utf-8'))
    cases = application['cases']
    queries = np.load(ROOT / '.codex-tmp/official-rwkv-20261028/int8-singleton-queries.npz')['vectors']
    if args.suite:
        suite = json.loads(args.suite.read_text(encoding='utf-8'))
        cases = [dict(kind='new_positive', group=group, query=query, evidence=evidence, targets=suite['target_groups'][group])
                 for group, query, evidence in suite['new_positive']]
        cases += [dict(kind='negative', query=query, targets=[]) for query in suite['negative']]
        benchmark_spec = importlib.util.spec_from_file_location('official_comparison', ROOT / 'scripts/compare-official-rwkv.py')
        benchmark = importlib.util.module_from_spec(benchmark_spec)
        benchmark_spec.loader.exec_module(benchmark)
        _, queries = benchmark.rust_vectors([case['query'] for case in cases], ROOT / 'assets/models/embedding-rwkv-tiny/model.onnx', args.threads, batch_size=1)
    report = dict(weights=str(weights), weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  layers=layers, template=TEMPLATE, threads=args.threads, precision='CPU FP32',
                  state_build=json.loads((output / 'state-build.json').read_text(encoding='utf-8')),
                  head_reference_parity='not validated against the original PyTorch WKV execution',
                  candidate_policy=dict(windows_per_document=args.windows, total_limit=args.windows * 3,
                                        selection='raw embedding cosine; no target/evidence oracle'),
                  results=[])
    for index, case in enumerate(cases):
        if args.queries and case['query'] not in args.queries:
            continue
        if args.query_limit and len(report['results']) >= args.query_limit:
            break
        scores = vectors @ queries[index]
        ordered = scores.argsort()[::-1]
        selected, counts = [], {}
        for passage in ordered:
            document = rows[passage][0]
            if counts.get(document, 0) < args.windows:
                counts[document] = counts.get(document, 0) + 1
                selected.append(int(passage))
            if len(selected) >= args.windows * 3:
                break
        pairs = [TEMPLATE.format(document=names[rows[passage][0]][:160] + '\n' + rows[passage][1], query=case['query']) for passage in selected]
        started = time.perf_counter()
        logits = []
        for start in range(0, len(pairs), 4):
            _, states = encoder.encode(pairs[start:start + 4])
            with torch.inference_mode():
                values = head(torch.from_numpy(states[layers].copy())).squeeze(-1)
            logits.extend(values.tolist())
        ranked = sorted(zip(selected, logits), key=lambda value: value[1], reverse=True)
        result = dict(**case, seconds=time.perf_counter() - started,
                      passages=[dict(index=passage, name=names[rows[passage][0]], logit=score, cosine=float(scores[passage]), text=rows[passage][1]) for passage, score in ranked])
        report['results'].append(result)
        save(output / 'result.json', report)
        print(json.dumps(dict(query=index + 1, kind=case['kind'], seconds=round(result['seconds'], 3),
            top=[(row['name'], round(row['logit'], 3)) for row in result['passages'][:3]]), ensure_ascii=True), flush=True)
    encoder.encoder.close()


if __name__ == '__main__':
    main()
