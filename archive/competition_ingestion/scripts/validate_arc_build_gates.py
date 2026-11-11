"""Verify ARC source manifest/content rejection before compiler invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.arc_build import build_solver, verified_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    args = parser.parse_args()
    verified = verified_sources(args.source_cache)
    checks = []
    with tempfile.TemporaryDirectory(prefix="sciona_arc_build_gates_") as temp:
        base = Path(temp)
        # Copy only the verified software allowlist, not arbitrary cache contents.
        for rel, data in verified.items():
            target = base / "cache" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        file = base / "cache/src/main.cpp"
        file.write_bytes(file.read_bytes() + b"\n// synthetic tamper\n")
        with patch("sciona.arc_build.subprocess.run", side_effect=AssertionError("Compiler must not run")):
            try:
                build_solver(base / "cache", base / "out")
            except ValueError as error:
                assert "source hash mismatch" in str(error)
            else:
                raise AssertionError("Modified source accepted")
        assert not (base / "out").exists()
        checks.append("modified_source_rejected_before_compilation_without_output")
        manifest_root = base / "manifest_root"
        (manifest_root / "docs/reviews").mkdir(parents=True)
        shutil.copyfile(ROOT / "docs/reviews/competition_arc_source_pins.json",
                        manifest_root / "docs/reviews/competition_arc_source_pins.json")
        file = manifest_root / "docs/reviews/competition_arc_source_pins.json"
        file.write_bytes(file.read_bytes() + b" ")
        with patch("sciona.arc_build.ROOT", manifest_root), patch("sciona.arc_build.subprocess.run", side_effect=AssertionError("Compiler must not run")):
            try:
                build_solver(args.source_cache, base / "out")
            except ValueError as error:
                assert "manifest hash mismatch" in str(error)
            else:
                raise AssertionError("Modified manifest accepted")
        assert not (base / "out").exists()
        checks.append("modified_manifest_rejected_before_compilation_without_output")
    report = {"format": "arc-build-gates.v1", "status": "passed", "checks": checks,
              "verified_source_files": len(verified),
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ["sciona/arc_build.py", "scripts/validate_arc_build_gates.py"]}}
    (ROOT / "docs/reviews/competition_arc_build_gates.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Both build tamper gates passed")


if __name__ == "__main__":
    main()
