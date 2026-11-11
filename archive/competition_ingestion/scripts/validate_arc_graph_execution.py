"""Execute the serialized ARC graph with synthetic inputs and reviewed binary."""
import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sciona.atoms.ml.arc_execution as provider
from sciona.arc_graph import build_arc_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = {"version": 1, "task": {"train": [
        {"input": [[0, 1], [1, 0]], "output": [[0, 1], [1, 0]]},
        {"input": [[2, 0], [0, 2]], "output": [[2, 0], [0, 2]]}],
        "test": [{"input": [[0, 3], [3, 0]]}, {"input": [[4, 0], [0, 4]]}]},
        "budget": {"seconds": 30, "memory_mib": 1024}}
    digest, nodes, edges = encode_execution_graph(build_arc_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    runner._ensure_atoms_imported()
    captured = {}

    def capture(directory, node, name, value):
        if node == "execute" and name == "out_result":
            captured["result"] = value

    with tempfile.TemporaryDirectory(prefix="sciona_arc_synthetic_graph_") as temp:
        with patch.dict(os.environ, {"SCIONA_ARC_BUILD_DIR": str(args.build_dir)}), \
             patch.object(runner, "RUNS_DIR", Path(temp)), \
             patch.object(runner, "save_intermediate_value", side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, "synthetic-arc", "case").execute({"payload": payload}, cdg=graph))
    assert status["status"] == "completed"
    result = json.loads(json.dumps(captured["result"], allow_nan=False))
    assert result["completed_all_runs"] and result["missing_test_indices"] == []
    assert len(result["attempts"]) == 8
    assert all(attempt["status"] == "success" for attempt in result["attempts"])
    assert result["search_arguments"] == [3, 23, 33, 4]
    assert all(item["input"] in [candidate["grid"] for candidate in predictions]
               for item, predictions in zip(payload["task"]["test"], result["predictions"], strict=True))
    assert provider.witness_arc_prepare({}) == {"kind": "ARC.Prepared"}
    assert provider.witness_arc_execute({"kind": "ARC.Prepared"}) == {"kind": "ARC.Result"}
    paths = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "sciona").glob("arc_*.py"))]
    paths += ["sciona/arc_build_identity.json", "scripts/validate_arc_graph_execution.py", "requirements/arc-execution.txt"]
    report = {"format": "arc-graph-execution.v1", "status": "passed", "approved": False,
              "serialized_graph_sha256": digest,
              "provider_sha256": hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
              "code_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
              "checks": {"actual_runner_nodes": 2, "full_source_search_runs": 8,
                         "synthetic_test_inputs": 2, "four_adaptive_phases": True,
                         "expected_predictions_retained": True, "strict_json_output": True,
                         "graph_codec_roundtrip": True, "provider_witness_contracts": True},
              "limits": "Synthetic tasks and a 30-second configured budget; no competition accuracy or served publication claim. Runtime grids captured in memory only."}
    (ROOT / "docs/reviews/competition_arc_graph_execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["checks"]))


if __name__ == "__main__":
    main()
