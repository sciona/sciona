"""Restore the pinned decoder's historical integer-division semantics.

Loads only the hash-checked source function, with its one index-division AST
operator changed from true division to floor division for modern PyTorch.
The caller installs the returned function in its isolated detector process.
"""
import ast
import hashlib
from pathlib import Path

import torch


BENCH_SHA256 = 'e5157e6cbee4b573a3e13e76743d7d9b9d7f4369ad03e7508913dc1c205be664'


def legacy_postprocess(source_path):
    raw = Path(source_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BENCH_SHA256:
        raise ValueError('decoder source content differs')
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == '_post_process')
    divisions = [n for n in ast.walk(node) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    if len(divisions) != 1 or ast.unparse(divisions[0]) != 'cls_topk_indices_all / config.num_classes':
        raise ValueError('unexpected decoder division contract')
    divisions[0].op = ast.FloorDiv()
    namespace = dict(torch=torch, MAX_DETECTION_POINTS=5000)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<pinned-wheat-integer-decoder>', 'exec'), namespace)
    return namespace['_post_process']
