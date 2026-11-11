"""Automated source/build/computation review for Tier 3 ARC publication."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ("source_pins", "source_execution", "source_contract_review", "build_review",
            "build_gates", "process_probe", "schedule_execution", "graph_execution",
            "runtime_gates", "environment")
SOURCE_VERSION = "4a5eaed3-fbc0-564f-b12b-120a76304e05"
SOURCE_HASH = "6be75e601582ede9f8122169f2ff4a763759a628a0631acae25b34a3fda826bf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    if not __debug__:
        raise RuntimeError("Review assertions must be enabled")
    reviews = ROOT / "docs/reviews"
    documents = {suffix: json.loads((reviews / f"competition_arc_{suffix}.json").read_text()) for suffix in EVIDENCE}
    for suffix, document in documents.items():
        for name, expected in document.get("code_sha256", {}).items():
            path = (ROOT / name).resolve()
            assert path.is_relative_to(ROOT) and sha(path) == expected, (suffix, name)
        if suffix not in {"source_pins", "source_execution", "source_contract_review"}:
            assert document["status"] == "passed", suffix
    from sciona.arc_build import verified_sources, PINS_SHA256, UNITS, PATCHES
    sources = verified_sources(Path("/private/tmp/sciona_arc_source"))
    pins = documents["source_pins"]
    assert pins["commit"] == "9407072659de1270358c2ba34c527785214dd68b"
    assert pins["repository"] == "https://github.com/top-quarks/ARC-solution"
    assert sources["LICENSE"] == (ROOT / "docs/licenses/ARC-solution-MIT.txt").read_bytes()
    original = documents["source_contract_review"]
    assert original["intake_version_id"] == SOURCE_VERSION and original["intake_content_hash"] == SOURCE_HASH
    assert original["source_commit"] == pins["commit"]
    assert original["source_pins_sha256"] == PINS_SHA256
    stage_ids = ["grid_translation", "dsl_construction", "tree_search", "program_validation", "inference"]
    assert [item["stage"] for item in original["corrections"]] == stage_ids
    first = documents["source_execution"]
    assert first["status"] == "preliminary_synthetic_execution_passed"
    assert first["validator_sha256"] == sha(ROOT / "scripts/validate_arc_source_execution.py")
    assert len(first["checks"]) == 12 and all(x["expected_in_top_three"] for x in first["checks"])
    build = documents["build_review"]
    assert build["validator_sha256"] == sha(ROOT / "scripts/validate_arc_build.py")
    assert build["second_release_byte_identical"] is True
    release, sanitized = build["release"], build["sanitized"]
    for manifest in [release, sanitized]:
        assert manifest["builder_sha256"] == sha(ROOT / "sciona/arc_build.py")
        assert manifest["source_pins_sha256"] == PINS_SHA256
        assert manifest["translation_units"] == UNITS and manifest["verified_source_files"] == 53
        assert {entry["path"] for entry in manifest["patches"]} == set(PATCHES)
        for entry in manifest["patches"]:
            before = sources[entry["path"]]
            after = before.decode()
            for old, new in PATCHES[entry["path"]]:
                after = after.replace(old, new)
            assert entry["before_sha256"] == hashlib.sha256(before).hexdigest()
            assert entry["after_sha256"] == hashlib.sha256(after.encode()).hexdigest()
    assert release["sanitizers"] is False and sanitized["sanitizers"] is True
    assert "-fsanitize=address,undefined" in sanitized["flags"] and "-fno-sanitize-recover=all" in sanitized["flags"]
    assert len(build["checks"]) == 30
    assert {(x["synthetic_case"], x["argument"]) for x in build["checks"]} == {
        (case, mode) for case in ["identity", "transpose", "crop", "color_change", "inconsistent_training"]
        for mode in [2, 3, 23, 33, 4, 12]}
    assert all(x["sanitizers_clean"] and x["release_sanitized_answers_and_scores_identical"] for x in build["checks"])
    assert len(documents["build_gates"]["checks"]) == 2
    assert set(documents["runtime_gates"]["checks"]) == {
        "binary_rejected_before_schedule", "manifest_rejected_before_schedule",
        "license_rejected_before_schedule", "missing_binary_rejected_before_schedule"}
    from sciona.arc_runtime import reviewed_binary, IDENTITY_SHA256
    identity = json.loads((ROOT / "sciona/arc_build_identity.json").read_text())
    assert identity["manifest"] == release
    assert sha(ROOT / "sciona/arc_build_identity.json") == IDENTITY_SHA256
    with patch.dict(os.environ, {"SCIONA_ARC_BUILD_DIR": os.environ.get("SCIONA_ARC_BUILD_DIR", "/private/tmp/sciona_arc_reviewed_build_v1")}):
        _, binary_hash = reviewed_binary()
    for suffix in ["process_probe", "schedule_execution"]:
        assert documents[suffix]["binary_sha256"] == binary_hash
    probe = documents["process_probe"]["checks"]
    assert len(probe) == 5
    assert all(x["all_four_source_modes_completed"] and x["identity_in_top_three"] for x in probe[:2])
    assert {x["forced_limit"] for x in probe[2:4]} == {"time_limit", "memory_limit"}
    assert all(x["no_candidates"] for x in probe[2:4]) and probe[4]["binary_hash_mismatch_rejected"]
    schedule = documents["schedule_execution"]
    assert schedule["source_schedule_sha256"] == hashlib.sha256(sources["safe_run.py"]).hexdigest()
    assert schedule["source_command_policy_differential_phases"] == 4
    assert schedule["successful_full_source_runs"] == 8 and len(schedule["attempts"]) == 8
    assert {(x["synthetic_test_index"], x["argument"]) for x in schedule["attempts"]} == {(i, m) for i in [0, 1] for m in [3, 23, 33, 4]}
    assert all(x["status"] == "success" for x in schedule["attempts"])
    for field in ["all_started_children_reaped", "both_synthetic_expected_predictions_retained",
                  "global_deadline_missing_predictions_explicit", "global_memory_missing_predictions_explicit"]:
        assert schedule[field] is True, field
    full = documents["graph_execution"]
    assert full["checks"] == {"actual_runner_nodes": 2, "full_source_search_runs": 8,
                              "synthetic_test_inputs": 2, "four_adaptive_phases": True,
                              "expected_predictions_retained": True, "strict_json_output": True,
                              "graph_codec_roundtrip": True, "provider_witness_contracts": True}
    runtime_files = {str(p.relative_to(ROOT)) for p in (ROOT / "sciona").glob("arc_*.py")}
    runtime_files.add("sciona/arc_build_identity.json")
    assert runtime_files <= set(full["code_sha256"])
    provider = ROOT.parent / "sciona-atoms-ml/src/sciona/atoms/ml/arc_execution.py"
    assert sha(provider) == full["provider_sha256"]
    from sciona.arc_graph import build_arc_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_arc_graph())[0] == full["serialized_graph_sha256"]
    dependencies = {}
    for line in (ROOT / "requirements/arc-execution.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            assert importlib.metadata.version(name) == version
            dependencies[name] = version
    assert documents["environment"]["direct_runtime_versions"] == dependencies
    assert documents["environment"]["tests_passed"] == 50
    stages = [
        ("grid_translation", "C++ signed-char Image, strict private grid boundary and reversible color normalization", ["source_contract_review", "source_execution", "graph_execution"]),
        ("dsl_construction", "Full pinned scalar/vector image DSL and cost-limited DAG expansion", ["build_review", "source_contract_review", "schedule_execution"]),
        ("tree_search", "Output-size deduction, cross-example piece extraction, greedy composition and outer-product deduction across 3/23/33/4 phases", ["source_contract_review", "build_review", "schedule_execution", "graph_execution"]),
        ("program_validation", "Source candidate training-agreement score with depth/piece penalty; partial agreement allowed, no saved-program claim", ["source_contract_review", "build_review"]),
        ("inference", "Test DAG evaluation, color reconstruction, scored top-three merging, explicit resource limits and missing predictions", ["process_probe", "schedule_execution", "graph_execution", "runtime_gates"])]
    auxiliary = ["scripts/review_arc_execution.py", "requirements/arc-execution.txt", "docs/licenses/ARC-solution-MIT.txt"]
    return {"format": "arc-semantic-review.v1", "review_source": "automated", "proposed_tier": 3,
            "verdict": "acceptable_with_limits", "source_version_id": SOURCE_VERSION, "source_hash": SOURCE_HASH,
            "source_commits": {"winning_solver": pins["commit"]},
            "serialized_graph_sha256": full["serialized_graph_sha256"], "provider_sha256": sha(provider),
            "provider_package_sha256": sha(ROOT.parent / "sciona-atoms-ml/pyproject.toml"),
            "build_identity_sha256": IDENTITY_SHA256,
            "intake_stage_mapping": [{"intake": s, "realization": r, "evidence": e} for s, r, e in stages],
            "dependencies": dependencies,
            "limitations": [
                "Automated Tier 3 Community reconstruction. No Tier 1 human certification or Tier 2 usage qualification; original intake remains draft.",
                "Original intake repository URL is superseded by the primary winner announcement linking top-quarks/ARC-solution. Source methods correct Python/A* and mandatory all-training-match claims.",
                "Full pinned C++ solver with two recorded portability patches and explicit signed-char compilation. Reviewed macOS ARM64 binary only; another platform/toolchain requires a new reviewed build identity.",
                "Same-toolchain release builds were byte-identical; 30 paired synthetic ASan/UBSan comparisons passed. These checks do not prove correctness for every input or historical GCC equivalence.",
                "Four adaptive phases, original concurrency and estimates preserved. Monotonic clock and 10ms RSS sampling differ from original 100ms monitor. Timed-out children are explicitly killed/reaped; missing baselines/outputs are exposed, not fabricated.",
                "Source nine-hour/95-percent time and memory defaults cap caller budgets; synthetic evidence used shorter explicit budgets. No nine-hour competition benchmark or historical accuracy claim.",
                "RSS is sampled process memory, not an instantaneous OS cap or parent-memory accounting. Zero observed baseline RSS yields a derived limit failure rather than silently increasing the budget.",
                "Strict rectangular integer grids, maximum 30x30 and explicit adapter caps of 20 training pairs/20 test inputs. Test labels rejected; each test is isolated. Predictions and runtime grids are private caller data.",
                "No implicit downloads or caller executable paths. A local reviewed build bundle is required; matcher supplies broader runner dependencies and clean-environment installation is unverified."],
            "finding": "Full source implementation, adaptive execution, build/source integrity and real serialized graph evidence support Tier 3 selection within the explicit runtime limits. Catalog publication and served verification remain separate gates.",
            "evidence_sha256": {f"competition_arc_{s}.json": sha(reviews / f"competition_arc_{s}.json") for s in EVIDENCE},
            "auxiliary_sha256": {p: sha(ROOT / p) for p in auxiliary}, "catalog_mutations": 0}


if __name__ == "__main__":
    result = audit()
    (ROOT / "docs/reviews/competition_arc_semantic_review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": result["verdict"], "proposed_tier": 3, "intake_stages": 5, "catalog_mutations": 0}))
