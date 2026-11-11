"""Synthetic full-source process probe; requires a previously verified build."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.arc_contract import validate_task, merge_answers
from sciona.arc_process import run_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    task = validate_task({"train": [
        {"input": [[0, 1], [1, 0]], "output": [[0, 1], [1, 0]]},
        {"input": [[2, 0], [0, 2]], "output": [[2, 0], [0, 2]]}],
        "test": [{"input": [[0, 3], [3, 0]]}, {"input": [[4, 0], [0, 4]]}]})
    checks = []
    for index in range(2):
        attempts = [run_once(args.binary, args.sha256, task, index, arg,
                             timeout_seconds=30, memory_mib=1024) for arg in [3, 23, 33, 4]]
        assert all(a.status == "success" for a in attempts)
        merged = merge_answers([a.candidates for a in attempts])
        assert task.test[index] in [a.grid for a in merged]
        checks.append({"synthetic_test_index": index, "all_four_source_modes_completed": True,
                       "identity_in_top_three": True,
                       "attempts": [{"argument": a.argument, "seconds": a.seconds,
                                     "peak_rss_mib": a.peak_rss_mib} for a in attempts]})
    for name, limits in [("time_limit", {"timeout_seconds": .001, "memory_mib": 1024}),
                         ("memory_limit", {"timeout_seconds": 30, "memory_mib": .001})]:
        a = run_once(args.binary, args.sha256, task, 0, 4, **limits)
        assert a.status == name and not a.candidates
        checks.append({"forced_limit": name, "status": a.status, "no_candidates": True})
    try:
        run_once(args.binary, "0" * 64, task, 0, 2, timeout_seconds=30, memory_mib=1024)
    except ValueError:
        checks.append({"binary_hash_mismatch_rejected": True})
    else:
        raise AssertionError("Mismatched binary hash accepted")
    report = {"format": "arc-process-probe.v1", "status": "passed", "binary_sha256": args.sha256,
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ["sciona/arc_process.py", "sciona/arc_contract.py",
                                           "scripts/validate_arc_process.py", "requirements/arc-execution.txt"]},
              "checks": checks,
              "limits": ["Local preliminary build, not yet a reproducible published runtime build.",
                         "Four source arguments executed serially; adaptive scheduler not yet implemented.",
                         "RSS sampled every 10ms; not an OS-enforced instantaneous memory ceiling.",
                         "Synthetic execution only; no competition accuracy or publication claim."]}
    (ROOT / "docs/reviews/competition_arc_process_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "checks": len(checks), "successful_full_source_runs": 8}))


if __name__ == "__main__":
    main()
