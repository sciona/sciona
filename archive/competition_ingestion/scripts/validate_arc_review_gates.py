"""Inject weaker ARC review evidence without modifying files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_arc_execution as review


def main():
    original = Path.read_text
    checks = []

    def failure(suffix, name, mutate):
        target = f"competition_arc_{suffix}.json"

        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path.name == target:
                document = json.loads(text)
                mutate(document)
                return json.dumps(document)
            return text

        with patch.object(Path, "read_text", changed):
            try:
                review.audit()
            except AssertionError:
                checks.append(name)
                return
        raise AssertionError("Weakened evidence accepted: " + name)

    failure("source_contract_review", "wrong_intake_hash", lambda d: d.__setitem__("intake_content_hash", "0" * 64))
    failure("source_pins", "wrong_source_commit", lambda d: d.__setitem__("commit", "0" * 40))
    failure("build_review", "sanitizer_failure", lambda d: d["checks"][0].__setitem__("sanitizers_clean", False))
    failure("build_review", "missing_search_mode", lambda d: d["checks"].pop())
    failure("schedule_execution", "incomplete_lifecycle", lambda d: d.__setitem__("successful_full_source_runs", 1))
    failure("schedule_execution", "children_not_reaped", lambda d: d.__setitem__("all_started_children_reaped", False))
    failure("graph_execution", "reduced_graph_execution", lambda d: d["checks"].__setitem__("full_source_search_runs", 1))
    failure("graph_execution", "altered_provider", lambda d: d.__setitem__("provider_sha256", "0" * 64))
    failure("graph_execution", "stale_runtime_code", lambda d: d["code_sha256"].__setitem__("sciona/arc_runtime.py", "0" * 64))
    failure("environment", "wrong_runtime_dependency", lambda d: d["direct_runtime_versions"].__setitem__("psutil", "0.0.0"))
    assert review.audit()["verdict"] == "acceptable_with_limits"
    paths = ["scripts/review_arc_execution.py", "scripts/validate_arc_review_gates.py"]
    result = {"format": "arc-review-negative-gates.v1", "result": "passed",
              "checks": {"injected_failures_rejected": len(checks), "unaltered_audit_passes": True},
              "rejected_scenarios": checks,
              "sha256": {p: hashlib.sha256((review.ROOT / p).read_bytes()).hexdigest() for p in paths},
              "limits": "Read-only review injections; transactional publication and served-catalog verification are separate."}
    (review.ROOT / "docs/reviews/competition_arc_review_gates.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"]))


if __name__ == "__main__":
    main()
