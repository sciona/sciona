"""Build the pinned MIT ARC implementation from a verified software-only cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PINS_SHA256 = "aebb89e39091fe6ac6f8e9d1718c088d90a1a87b0efd0adfe8c88c39176b68fc"
UNITS = "read core_functions image_functions image_functions2 visu normalize tasks runner score load evals brute2 deduce_op pieces compose2 brute_size efficient main".split()
PATCHES = {
    "src/read.cpp": [("<experimental/filesystem>", "<filesystem>"),
                     ("experimental::filesystem::", "filesystem::")],
    "headers/precompiled_stl.hpp": [("#include <vector>", "#include <map>\n#include <chrono>\n#include <vector>")],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_sources(cache: Path) -> dict[str, bytes]:
    raw = (ROOT / "docs/reviews/competition_arc_source_pins.json").read_bytes()
    if digest(raw) != PINS_SHA256:
        raise ValueError("ARC source manifest hash mismatch")
    result = {}
    for entry in json.loads(raw)["files"]:
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts or path.as_posix() in result:
            raise ValueError("Invalid source manifest entry")
        data = (cache / path).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest(data) != entry["sha256"] or blob != entry["git_blob_sha1"]:
            raise ValueError(f"ARC source hash mismatch: {path}")
        result[path.as_posix()] = data
    return result


def build_solver(cache: Path, output: Path, *, compiler: str = "clang++", sanitize: bool = False) -> dict:
    """Produce a fresh build bundle. Does not execute source launchers or fetch data."""
    sources = verified_sources(cache)
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise ValueError("C++ compiler unavailable")
    version = subprocess.run([compiler_path, "--version"], check=True, capture_output=True, text=True).stdout
    flags = ["-std=c++17", "-O1" if sanitize else "-O2", "-fsigned-char", "-Iheaders"]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
    patches = []
    with tempfile.TemporaryDirectory(prefix="sciona_arc_build_") as temp:
        work = Path(temp)
        for rel, data in sources.items():
            if Path(rel).parts[0] not in {"src", "headers"}:
                continue
            target = work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for rel, replacements in PATCHES.items():
            before = (work / rel).read_bytes()
            text = before.decode()
            for old, new in replacements:
                if old not in text:
                    raise ValueError("Missing source patch anchor")
                text = text.replace(old, new)
            after = text.encode()
            (work / rel).write_bytes(after)
            patches.append({"path": rel, "before_sha256": digest(before), "after_sha256": digest(after),
                            "replacements": replacements})
        process = subprocess.run([compiler_path, *flags, *[f"src/{name}.cpp" for name in UNITS], "-o", "solver"],
                                 cwd=work, capture_output=True, text=True, timeout=180)
        if process.returncode:
            raise RuntimeError("ARC compiler failed: " + process.stderr[-6000:])
        binary = (work / "solver").read_bytes()
        manifest = {"format": "arc-build.v1", "source_pins_sha256": PINS_SHA256,
                    "builder_sha256": digest(Path(__file__).read_bytes()),
                    "compiler_version": version.strip(), "flags": flags, "sanitizers": sanitize,
                    "translation_units": UNITS, "verified_source_files": len(sources),
                    "patches": patches, "binary_sha256": digest(binary),
                    "compiler_warning_count": process.stderr.count("warning:")}
        # Never overwrite a previously reviewed build, including symlink targets.
        output.mkdir(parents=True, exist_ok=False)
        (output / "solver").write_bytes(binary)
        (output / "solver").chmod(0o700)
        (output / "LICENSE").write_bytes(sources["LICENSE"])
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
