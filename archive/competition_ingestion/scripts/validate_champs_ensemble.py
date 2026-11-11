"""Compare adapter to pinned original ensemble using exclusively synthetic I/O."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciona.champs_ensemble import MODEL_ORDER, COUPLING_TYPES, blend_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo / "docs/reviews/competition_champs_source_pins.json").read_text())
    hashes = {p["software_path"]: p["sha256"] for p in pins["pins"]}
    source = (args.source / "src/predictor.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == hashes["src/predictor.py"]
    cfg_bytes = (args.source / "config/models.json").read_bytes()
    assert hashlib.sha256(cfg_bytes).hexdigest() == hashes["config/models.json"]
    cfg = json.loads(cfg_bytes)
    assert tuple(cfg[name + "_dir"] for name in cfg["names"]) == MODEL_ORDER
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in ("select_models", "ensemble")]
    assert len(nodes) == 2

    class Compatibility(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr in ("int", "bool"):
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            return self.generic_visit(node)

        def visit_Assert(self, node):
            # The original's fixed competition cardinality is replaced by the
            # synthetic case count, without altering the selection/reduction.
            assert ast.unparse(node.test) == "answer_count == 2505542"
            node.test.comparators[0] = ast.Name(id="case_count", ctx=ast.Load())
            return node

    module = ast.fix_missing_locations(Compatibility().visit(ast.Module(body=nodes, type_ignores=[])))
    cases = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        matrix = rng.normal(size=(13, 40))
        if seed == 0:
            matrix[:] = 1.0
        kinds = [COUPLING_TYPES[i % 8] for i in range(40)]
        csv = "a,b,c,d,type\n" + "\n".join("0,0,0,0," + t for t in kinds)
        ns = {"np": np, "os": os, "root": "", "settings": {"RAW_DATA_DIR": ""},
              "median_mean_counts": [5]*8, "case_count": 40,
              "open": lambda *a, **k: io.StringIO(csv),
              "load_submission": lambda name: np.column_stack((np.arange(40), matrix[cfg["names"].index(name)]))}
        exec(compile(module, "<pinned-ensemble>", "exec"), ns)
        with contextlib.redirect_stdout(io.StringIO()):
            _, expected = ns["ensemble"](cfg["names"])
        predictions = {m: {f"synthetic-{i}": float(matrix[j, i]) for i in rng.permutation(40)}
                       for j, m in enumerate(MODEL_ORDER)}
        actual = blend_predictions(predictions, {f"synthetic-{i}": t for i, t in enumerate(kinds)})
        np.testing.assert_array_equal(list(actual.values()), expected)
        cases.append({"seed": seed, "records": 40, "exact_match": True})
    args.output.write_text(json.dumps({"cases": cases, "source_commit": pins["commit"],
        "scope": "All eight coupling types and thirteen models; ties and permuted mapping order. Original functions executed with synthetic I/O, modern numpy aliases and synthetic cardinality assertion.",
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    print("12 source comparisons passed (480 synthetic predictions)")


if __name__ == "__main__":
    main()
