"""Clean ARC builds, reproducibility comparison and synthetic sanitizer checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.arc_build import build_solver
from sciona.arc_contract import parse_answers, validate_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = build_solver(args.source_cache, args.output)
    print("Release build complete", flush=True)
    cases = {
        "identity": ([([[0, 1], [1, 0]], [[0, 1], [1, 0]]),
                      ([[2, 0], [0, 2]], [[2, 0], [0, 2]])], [[0, 3], [3, 0]]),
        "transpose": ([([[1, 0, 0], [0, 1, 0]], [[1, 0], [0, 1], [0, 0]]),
                       ([[0, 2, 0], [0, 0, 2]], [[0, 0], [2, 0], [0, 2]])], [[3, 3, 0], [0, 0, 3]]),
        "crop": ([([[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[1]]),
                  ([[0, 2, 2], [0, 0, 0]], [[2, 2]])], [[0, 0, 0], [0, 3, 0], [0, 3, 0]]),
        "color_change": ([([[0, 1], [1, 0]], [[0, 2], [2, 0]]),
                          ([[1, 0, 1]], [[2, 0, 2]])], [[0, 1, 1], [1, 0, 0]]),
        "inconsistent_training": ([([[1]], [[2]]), ([[1]], [[3]])], [[1]]),
    }
    checks = []
    with tempfile.TemporaryDirectory(prefix="sciona_arc_build_review_") as temp:
        base = Path(temp)
        second = build_solver(args.source_cache, base / "release_again")
        identical = release["binary_sha256"] == second["binary_sha256"]
        print("Second release byte-identical:", identical, flush=True)
        sanitized = build_solver(args.source_cache, base / "sanitized", sanitize=True)
        print("Sanitizer build complete", flush=True)
        for alias, (training, test) in cases.items():
            task = validate_task({"train": [{"input": x, "output": y} for x, y in training],
                                  "test": [{"input": test}]})
            for argument in [2, 3, 23, 33, 4, 12]:
                outputs = []
                for flavor, binary in [("release", args.output.resolve() / "solver"),
                                       ("sanitized", base / "sanitized/solver")]:
                    work = base / f"{alias}_{argument}_{flavor}"
                    (work / "dataset/evaluation").mkdir(parents=True)
                    (work / "output").mkdir()
                    (work / "dataset/evaluation/00000000.json").write_text(json.dumps(task.source_task(0)))
                    result = subprocess.run([str(binary), "0", str(argument)], cwd=work,
                                            capture_output=True, text=True, timeout=60)
                    if result.returncode or "runtime error:" in result.stderr or "ERROR: AddressSanitizer" in result.stderr:
                        raise RuntimeError(f"{alias}/{argument}/{flavor}: {result.stderr[-5000:]}")
                    text = (work / f"output/answer_0_{argument}.csv").read_text()
                    parse_answers(text)
                    outputs.append(text)
                if outputs[0] != outputs[1]:
                    raise AssertionError(f"Sanitized/release output mismatch for {alias}/{argument}")
                checks.append({"synthetic_case": alias, "argument": argument, "sanitizers_clean": True,
                               "release_sanitized_answers_and_scores_identical": True})
            print(alias, "six modes passed", flush=True)
    report = {"format": "arc-build-review.v1", "status": "passed", "release": release,
              "second_release_byte_identical": identical, "sanitized": sanitized, "checks": checks,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": ["Sanitizer evidence covers these synthetic tasks only, not all possible inputs.",
                         "Reproducibility comparison uses the same installed compiler and platform.",
                         "No competition grids, historical GCC equivalence or competition accuracy evaluated."]}
    (ROOT / "docs/reviews/competition_arc_build_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Recorded ARC build review", flush=True)


if __name__ == "__main__":
    main()
