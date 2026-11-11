"""Compare scheduler policy with pinned source and run synthetic full solver."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.arc_contract import validate_task
from sciona.arc_process import Attempt
from sciona.arc_schedule import commands_for_phase, run_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((ROOT / "docs/reviews/competition_arc_source_pins.json").read_text())
    expected = next(x["sha256"] for x in pins["files"] if x["path"] == "safe_run.py")
    source = (args.source_cache / "safe_run.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == expected
    tree = ast.parse(source)
    # Execute only the Command class and four phase command-construction blocks.
    # Imports, filesystem operations, subprocess runner and submission are excluded.
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Command"]
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "depth3")
    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "stats4")
    selected += tree.body[start:end + 1]
    recorded = []

    def capture(commands, threads):
        recorded.append((sorted(commands), threads))
        return {command.cmd: (0, [2, 1, 1][int(command.cmd.split()[1])],
                              [8, 4, 5][int(command.cmd.split()[1])]) for command in commands}

    namespace = {"TIME_LIMIT": 100, "MEMORY_LIMIT": 200, "ntasks": 3, "runAll": capture}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "pinned_arc_command_policy", "exec"), namespace)
    baseline = {i: Attempt("success", 3, t, m, ()) for i, (t, m) in enumerate([(2, 8), (1, 4), (1, 5)])}
    for mode, (source_commands, source_threads) in zip([3, 23, 33, 4], recorded, strict=True):
        ours, threads = commands_for_phase(mode, 3, baseline, 100, 200)
        assert threads == source_threads
        assert [(x.test_index, x.argument, x.expected_seconds, x.expected_memory_mib, x.slack) for x in ours] == [
            (int(x.cmd.split()[1]), int(x.cmd.split()[2]), x.time, x.mem, x.slack) for x in source_commands]
    task = validate_task({"train": [
        {"input": [[0, 1], [1, 0]], "output": [[0, 1], [1, 0]]},
        {"input": [[2, 0], [0, 2]], "output": [[2, 0], [0, 2]]}],
        "test": [{"input": [[0, 3], [3, 0]]}, {"input": [[4, 0], [0, 4]]}]})
    children = []
    original = subprocess.Popen

    def tracked(*values, **kwargs):
        process = original(*values, **kwargs)
        children.append(process)
        return process

    with patch("sciona.arc_process.subprocess.Popen", side_effect=tracked):
        full = run_schedule(args.binary, args.sha256, task, seconds=30, memory_mib=1024)
        assert full.completed_all_runs and not full.missing_test_indices
        assert all(grid in [candidate.grid for candidate in predictions]
                   for grid, predictions in zip(task.test, full.predictions))
        deadline = run_schedule(args.binary, args.sha256, task, seconds=.001, memory_mib=1024)
        assert not deadline.completed_all_runs and deadline.missing_test_indices == (0, 1)
        memory = run_schedule(args.binary, args.sha256, task, seconds=30, memory_mib=.001)
        assert not memory.completed_all_runs and memory.missing_test_indices == (0, 1)
        assert all(process.poll() is not None for process in children)
    report = {"format": "arc-schedule-execution.v1", "status": "passed",
              "binary_sha256": args.sha256, "source_schedule_sha256": expected,
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
                  "sciona/arc_contract.py", "sciona/arc_process.py", "sciona/arc_schedule.py",
                  "scripts/validate_arc_schedule.py", "tests/test_arc_schedule.py"]},
              "source_command_policy_differential_phases": 4,
              "successful_full_source_runs": len(full.attempts),
              "both_synthetic_expected_predictions_retained": True,
              "global_deadline_missing_predictions_explicit": True,
              "global_memory_missing_predictions_explicit": True,
              "all_started_children_reaped": True, "started_process_count": len(children),
              "attempts": [{"synthetic_test_index": i, "argument": a.argument, "status": a.status,
                            "seconds": a.seconds, "peak_rss_mib": a.peak_rss_mib} for i, a in full.attempts],
              "limits": ["Thirty-second configured synthetic budget; no nine-hour competition benchmark.",
                         "Source-equivalent command estimates and phases, not identical wall-clock resource decisions.",
                         "Ten-millisecond RSS sampling; transient peaks and parent overhead are not an OS-enforced cap.",
                         "Build reproducibility, sanitizer review, provider/CDG and publication gates remain."]}
    (ROOT / "docs/reviews/competition_arc_schedule_execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Passed four source policy comparisons, eight full solver runs, global limits and child reaping.")


if __name__ == "__main__":
    main()
