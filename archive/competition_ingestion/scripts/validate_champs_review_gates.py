"""Inject weaker CHAMPS review evidence without modifying files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_champs_execution as review


def main():
    original = Path.read_text
    checks = []

    def failure(suffix, name, mutate):
        target = f"competition_champs_{suffix}.json"

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

    failure("source_pins", "wrong_source_commit", lambda d:d.__setitem__("commit","0"*40))
    failure("source_pins", "wrong_license", lambda d:d["license"].__setitem__("spdx_id","UNKNOWN"))
    failure("ensemble_execution", "wrong_source_reduction", lambda d:d["cases"][0].__setitem__("exact_match",False))
    failure("loss_execution", "wrong_gradient", lambda d:d["results"][0].__setitem__("scalar_loss_and_gradient_match",False))
    failure("prediction_execution", "missing_source_rounding", lambda d:d.__setitem__("six_decimal_source_boundary",False))
    failure("training_runtime_streaming_execution", "missing_optimizer_schedule", lambda d:d["results"].pop())
    failure("lifecycle_execution", "reduced_model_ensemble", lambda d:d["results"].pop())
    failure("graph_execution", "reduced_graph_execution", lambda d:d["checks"].__setitem__("full_model_lifecycles",1))
    failure("graph_execution", "altered_provider", lambda d:d.__setitem__("provider_sha256","0"*64))
    failure("environment", "wrong_runtime_dependency", lambda d:d["direct_runtime_versions"].__setitem__("torch","0.0.0"))
    assert review.audit()["verdict"] == "acceptable_with_limits"
    paths = ["scripts/review_champs_execution.py", "scripts/validate_champs_review_gates.py"]
    result = {"format": "champs-review-negative-gates.v1", "result": "passed",
              "checks": {"injected_failures_rejected": len(checks), "unaltered_audit_passes": True},
              "rejected_scenarios": checks,
              "sha256": {p: hashlib.sha256((review.ROOT / p).read_bytes()).hexdigest() for p in paths},
              "limits": "Read-only review injections; transactional publication and served-catalog verification are separate."}
    (review.ROOT / "docs/reviews/competition_champs_review_gates.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"]))


if __name__ == "__main__":
    main()
