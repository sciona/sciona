"""Reject altered ARC deployment bundles before any solver launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.arc_runtime import prepare, execute, reviewed_binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    with patch.dict(os.environ, {"SCIONA_ARC_BUILD_DIR": str(args.build_dir)}):
        reviewed_binary()
    prepared = prepare({"version": 1, "task": {"train": [{"input": [[1]], "output": [[1]]}],
                                              "test": [{"input": [[2]]}]},
                        "budget": {"seconds": 30, "memory_mib": 1024}})
    checks = []
    with tempfile.TemporaryDirectory(prefix="sciona_arc_bundle_gates_") as temp:
        base = Path(temp)
        for scenario in ["binary", "manifest", "license", "missing_binary"]:
            bundle = base / scenario
            bundle.mkdir()
            for name in ["solver", "LICENSE", "manifest.json"]:
                if scenario == "missing_binary" and name == "solver":
                    continue
                shutil.copyfile(args.build_dir / name, bundle / name)
            if scenario == "binary":
                with (bundle / "solver").open("ab") as file:
                    file.write(b"synthetic-tamper")
            elif scenario == "manifest":
                path = bundle / "manifest.json"
                value = json.loads(path.read_text())
                value["flags"] = ["-O0"]
                path.write_text(json.dumps(value))
            elif scenario == "license":
                (bundle / "LICENSE").write_text("synthetic replacement")
            with patch.dict(os.environ, {"SCIONA_ARC_BUILD_DIR": str(bundle)}), \
                 patch("sciona.arc_runtime.run_schedule", side_effect=AssertionError("Solver must not launch")):
                try:
                    execute(prepared)
                except (ValueError, RuntimeError):
                    checks.append(scenario + "_rejected_before_schedule")
                else:
                    raise AssertionError("Altered bundle accepted")
    report = {"format": "arc-runtime-gates.v1", "status": "passed", "checks": checks,
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
                  "sciona/arc_runtime.py", "sciona/arc_build_identity.json", "scripts/validate_arc_runtime_gates.py"]}}
    (ROOT / "docs/reviews/competition_arc_runtime_gates.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Four deployment bundle rejection gates passed")


if __name__ == "__main__":
    main()
