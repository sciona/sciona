"""Extractor: export verified source to compiled artifacts and FFI bindings."""

from __future__ import annotations

import asyncio
import ast
import copy
from enum import Enum
from pathlib import Path
import re

from sciona.config import AgeomConfig
from sciona.synthesizer.cargo_template import (
    generate_build_rs,
    generate_cargo_toml,
    generate_lib_rs,
)
from sciona.synthesizer.certificate import generate_certificate, save_certificate
from sciona.synthesizer.lakefile_template import generate_lakefile
from sciona.synthesizer.models import ExportBundle, SkeletonFile, SynthesisResult


class ExportTarget(str, Enum):
    """Supported export targets."""

    LEAN_LIB = "lean-lib"
    COQ_LIB = "coq-lib"
    RUST_FFI = "rust-ffi"
    C_HEADER = "c-header"
    PYTHON_PKG = "python-pkg"


def _prepare_python_package_source(
    source: str,
    *,
    required_modules: list[str] | None = None,
) -> str:
    """Tweak exported Python source for package-level mypy validation."""
    lines = source.splitlines()
    if not lines:
        return source

    result: list[str] = []
    inserted_np_alias = False
    needs_np_alias = "np." in source and "import numpy as np" not in source
    required_imports = [
        f"import {module}"
        for module in required_modules or []
        if module and f"import {module}" not in source and f"from {module} import" not in source
    ]
    pending_imports = required_imports.copy()

    for line in lines:
        if needs_np_alias and line == "import numpy":
            result.append("import numpy as np")
            inserted_np_alias = True

        result.append(line)

    if needs_np_alias and not inserted_np_alias:
        for idx, line in enumerate(result):
            if line.startswith("from __future__ import"):
                continue
            result.insert(idx, "import numpy as np")
            break
    if pending_imports:
        insert_idx = 0
        for idx, line in enumerate(result):
            if line.startswith("from __future__ import") or line.startswith("import ") or line.startswith("from "):
                insert_idx = idx + 1
                continue
            break
        for item in pending_imports:
            result.insert(insert_idx, item)
            insert_idx += 1

    prepared = "\n".join(result)
    if source.endswith("\n"):
        prepared += "\n"
    return prepared


def _collect_dotted_call_modules(source: str) -> list[str]:
    """Infer dotted module imports needed for qualified function calls."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []

    imported_roots: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.asname or alias.name.split(".", 1)[0]).strip()
                if root:
                    imported_roots.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".", 1)[0].strip())
            for alias in node.names:
                root = (alias.asname or alias.name).split(".", 1)[0].strip()
                if root:
                    imported_roots.add(root)

    modules: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        parts: list[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if not isinstance(func, ast.Name):
            continue
        parts.append(func.id)
        if func.id not in imported_roots:
            continue
        qualname = ".".join(reversed(parts))
        if qualname.count(".") < 2:
            continue
        modules.add(qualname.rsplit(".", 1)[0])
    return sorted(modules)


def _infer_function_node_ids(source: str) -> dict[str, str]:
    """Infer function-to-node ids from assembler comments in generated Python."""
    node_ids: dict[str, str] = {}
    pending: str | None = None
    comment_rx = re.compile(r"# (?:Node|Composition): .* \(([^)]+)\)")
    def_rx = re.compile(r"def ([A-Za-z_][A-Za-z0-9_]*)\(")
    for line in source.splitlines():
        stripped = line.strip()
        comment_match = comment_rx.match(stripped)
        if comment_match:
            pending = comment_match.group(1)
            continue
        def_match = def_rx.match(stripped)
        if def_match and pending is not None:
            node_ids[def_match.group(1)] = pending
            pending = None
    return node_ids


def _telemetry_helper_source() -> str:
    return """\
import json
import time
import tracemalloc
from sciona.principal.runtime_context import summarize_named_value as _sciona_summarize_named_value

_SCIONA_TRACE_PATH = 'trace.jsonl'


def _sciona_summarize_outputs(output_names, result):
    names = [str(name) for name in output_names or () if name]
    if not names:
        return {}
    if len(names) == 1:
        values = [result]
    elif isinstance(result, (tuple, list)):
        values = list(result)
    else:
        values = [result]
    summaries = {}
    for index, name in enumerate(names):
        if index >= len(values):
            break
        try:
            summaries[name] = _sciona_summarize_named_value(name, values[index])
        except Exception:
            continue
    return summaries


def _sciona_probe(node_id: str, fn, output_names=(), guard_fn=None):
    result = None
    guard_info = None
    aborted_error = None
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        if guard_fn is not None:
            guard_info = guard_fn()
        result = fn()
    except Exception as _exc:
        guard_info = getattr(_exc, "guard_summary", guard_info)
        aborted_error = f"{type(_exc).__name__}: {_exc}"
        raise
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record = {
            "node_id": node_id,
            "execution_time_ms": elapsed_ms,
            "peak_memory_bytes": peak,
        }
        if guard_info is not None:
            record["confidence_guard"] = guard_info
        if aborted_error is not None:
            record["aborted_error"] = aborted_error
        summaries = _sciona_summarize_outputs(output_names, result)
        if summaries:
            record["output_summaries"] = summaries
        with open(_SCIONA_TRACE_PATH, "a") as handle:
            handle.write(json.dumps(record) + "\\n")
    return result
"""


def _build_wrapper_function(
    original: ast.FunctionDef,
    inner_name: str,
    node_id: str,
) -> ast.FunctionDef:
    wrapper = copy.deepcopy(original)
    wrapper.decorator_list = []
    doc = ast.get_docstring(original)
    body: list[ast.stmt] = []
    if doc is not None:
        body.append(ast.Expr(value=ast.Constant(value=doc)))

    call = ast.Call(func=ast.Name(id=inner_name, ctx=ast.Load()), args=[], keywords=[])
    for arg in wrapper.args.posonlyargs + wrapper.args.args:
        call.args.append(ast.Name(id=arg.arg, ctx=ast.Load()))
    if wrapper.args.vararg is not None:
        call.args.append(
            ast.Starred(value=ast.Name(id=wrapper.args.vararg.arg, ctx=ast.Load()), ctx=ast.Load())
        )
    for arg in wrapper.args.kwonlyargs:
        call.keywords.append(ast.keyword(arg=arg.arg, value=ast.Name(id=arg.arg, ctx=ast.Load())))
    if wrapper.args.kwarg is not None:
        call.keywords.append(
            ast.keyword(arg=None, value=ast.Name(id=wrapper.args.kwarg.arg, ctx=ast.Load()))
        )

    output_names = _infer_output_names(original)
    probe_call = ast.Call(
        func=ast.Name(id="_sciona_probe", ctx=ast.Load()),
        args=[
            ast.Constant(value=node_id),
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=call,
            ),
            ast.Tuple(
                elts=[ast.Constant(value=name) for name in output_names],
                ctx=ast.Load(),
            ),
        ],
        keywords=[],
    )
    body.append(ast.Return(value=probe_call))
    wrapper.body = body
    return wrapper


def _infer_output_names(original: ast.FunctionDef) -> list[str]:
    """Infer stable semantic output names for exported wrapper telemetry."""
    name = original.name.lower()
    if any(token in name for token in ("detect_peaks", "detect_peak", "find_peaks", "find_events")):
        return ["events"]
    if any(token in name for token in ("event_rate", "heart_rate", "compute_rate", "measure_rate")):
        return ["rate"]
    if any(token in name for token in ("filter_signal", "condition_signal", "bandpass", "denoise")):
        return ["signal"]
    if any(token in name for token in ("signal_quality", "quality", "snr")):
        return ["quality"]
    if any(token in name for token in ("outlier", "peak_correction", "correct_peaks")):
        return ["events"]
    return []


def _ensure_python_export_telemetry(source: str) -> str:
    """Ensure exported Python source emits trace.jsonl records for top-level functions."""
    if "_SCIONA_TRACE_PATH" in source and "def _sciona_probe" in source:
        return source

    module = ast.parse(source)
    node_ids = _infer_function_node_ids(source)
    helper_module = ast.parse(_telemetry_helper_source())

    new_body: list[ast.stmt] = []
    inserted_helper = False
    helper_insert_at = 0
    for idx, stmt in enumerate(module.body):
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            helper_insert_at = idx + 1
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            helper_insert_at = idx + 1
        else:
            break

    for idx, stmt in enumerate(module.body):
        if not inserted_helper and idx == helper_insert_at:
            new_body.extend(copy.deepcopy(helper_module.body))
            inserted_helper = True

        if isinstance(stmt, ast.FunctionDef) and not stmt.name.startswith("_"):
            inner = copy.deepcopy(stmt)
            inner_name = f"_sciona_inner_{stmt.name}"
            inner.name = inner_name
            wrapper = _build_wrapper_function(
                stmt,
                inner_name=inner_name,
                node_id=node_ids.get(stmt.name, stmt.name),
            )
            new_body.append(inner)
            new_body.append(wrapper)
        else:
            new_body.append(stmt)

    if not inserted_helper:
        new_body = copy.deepcopy(helper_module.body) + new_body

    instrumented = ast.Module(body=new_body, type_ignores=[])
    ast.fix_missing_locations(instrumented)
    rendered = ast.unparse(instrumented)
    if source.endswith("\n"):
        rendered += "\n"
    return rendered


def _discover_python_entrypoints(source: str) -> tuple[list[str], str | None]:
    """Return callable entrypoints from exported Python source."""
    module = ast.parse(source)
    names = [
        stmt.name
        for stmt in module.body
        if isinstance(stmt, ast.FunctionDef) and not stmt.name.startswith("_")
    ]
    compositions = [name for name in names if name.endswith("_composition")]
    candidates = compositions or names
    default = candidates[0] if candidates else None
    return candidates, default


class Extractor:
    """Exports verified source into compiled artifacts and FFI bindings."""

    def __init__(self, config: AgeomConfig) -> None:
        self._config = config

    async def extract(
        self,
        synthesis_result: SynthesisResult,
        target: ExportTarget,
        output_dir: Path,
    ) -> ExportBundle:
        """Build the export bundle for the given target."""
        output_dir.mkdir(parents=True, exist_ok=True)
        skeleton = synthesis_result.skeleton
        errors: list[str] = []

        # Write verified source
        src_dir = output_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        if skeleton.prover == "lean4":
            ext = ".lean"
        elif skeleton.prover == "python":
            ext = ".py"
        else:
            ext = ".v"
        source_path = src_dir / f"Verified{ext}"
        source_path.write_text(skeleton.source_code)

        compiled_artifact: Path | None = None
        executable_artifact: Path | None = None
        ffi_files: list[Path] = []

        if target == ExportTarget.LEAN_LIB:
            compiled_artifact, build_errors = await self._build_lean(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)

        elif target == ExportTarget.COQ_LIB:
            compiled_artifact, build_errors = await self._build_coq(
                source_path, output_dir
            )
            errors.extend(build_errors)

        elif target == ExportTarget.RUST_FFI:
            compiled_artifact, build_errors = await self._build_lean(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)
            ffi_files = self._generate_rust_ffi(skeleton, output_dir)

        elif target == ExportTarget.C_HEADER:
            compiled_artifact, build_errors = await self._build_lean(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)
            ffi_files = self._generate_c_header(skeleton, output_dir)

        elif target == ExportTarget.PYTHON_PKG:
            compiled_artifact, executable_artifact, build_errors = await self._build_python(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)

        # Generate certificate
        goal = skeleton.metadata.get("goal", "")
        cert = generate_certificate(
            source_path=source_path,
            artifact_path=compiled_artifact,
            skeleton=skeleton,
            prover_version=(
                self._config.lean_toolchain
                if skeleton.prover == "lean4"
                else "python" if skeleton.prover == "python" else "coq"
            ),
            goal=goal,
        )
        cert_path = output_dir / "certificate.json"
        save_certificate(cert, cert_path)

        return ExportBundle(
            target=target.value,
            output_dir=output_dir,
            source_path=source_path,
            compiled_artifact=compiled_artifact,
            executable_artifact=executable_artifact,
            ffi_files=ffi_files,
            certificate=cert,
            errors=errors,
        )

    async def _build_lean(
        self,
        skeleton: SkeletonFile,
        source_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, list[str]]:
        """Generate lakefile and run lake build."""
        errors: list[str] = []

        # Determine library name from source filename
        lib_name = source_path.stem

        # Write lakefile.lean
        lakefile_content = generate_lakefile(
            name=lib_name,
            lean_version=self._config.lean_toolchain,
        )
        lakefile_path = output_dir / "lakefile.lean"
        lakefile_path.write_text(lakefile_content)

        # Write lean-toolchain
        toolchain_path = output_dir / "lean-toolchain"
        toolchain_path.write_text(self._config.lean_toolchain + "\n")

        # Run lake build
        lake_bin = self._config.lean_lake_path
        try:
            proc = await asyncio.create_subprocess_exec(
                lake_bin,
                "build",
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                errors.append(
                    f"lake build failed (exit {proc.returncode}): {stderr.decode()[:500]}"
                )
                return None, errors
        except FileNotFoundError:
            errors.append(
                f"lake binary not found at '{lake_bin}' — skipping compilation"
            )
            return None, errors

        # Find .olean artifact
        build_dir = output_dir / ".lake" / "build" / "lib"
        olean_candidates = (
            list(build_dir.rglob("*.olean")) if build_dir.exists() else []
        )
        artifact = olean_candidates[0] if olean_candidates else None

        return artifact, errors

    async def _build_coq(
        self,
        source_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, list[str]]:
        """Run coqc to compile the source."""
        errors: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "coqc",
                str(source_path),
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                errors.append(
                    f"coqc failed (exit {proc.returncode}): {stderr.decode()[:500]}"
                )
                return None, errors
        except FileNotFoundError:
            errors.append("coqc not found — skipping compilation")
            return None, errors

        vo_path = source_path.with_suffix(".vo")
        artifact = vo_path if vo_path.exists() else None

        return artifact, errors

    async def _build_python(
        self,
        skeleton: SkeletonFile,
        source_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, Path | None, list[str]]:
        """Generate Python package and run mypy to validate."""
        from sciona.synthesizer.python_template import (
            generate_init_py,
            generate_pipeline_py,
            generate_pyproject_toml,
            generate_runner_py,
        )

        errors: list[str] = []
        package_name = skeleton.metadata.get("name", "verified_pkg")

        # Collect dependencies from imports in source
        dependencies = ["numpy", "scipy"]

        # Generate pyproject.toml
        pyproject_content = generate_pyproject_toml(package_name, dependencies)
        pyproject_path = output_dir / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)

        # Generate package structure
        pkg_dir = output_dir / "src" / package_name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # __init__.py
        exports = [u.name for u in skeleton.units]
        init_content = generate_init_py(package_name, exports)
        (pkg_dir / "__init__.py").write_text(init_content)

        # atoms.py — the verified source
        instrumented_source = _ensure_python_export_telemetry(skeleton.source_code)
        required_modules = sorted(
            {
                unit.declaration_name.rsplit(".", 1)[0]
                for unit in skeleton.units
                if "." in unit.declaration_name
            }
        )
        required_modules.extend(
            module
            for module in _collect_dotted_call_modules(instrumented_source)
            if module not in required_modules
        )
        prepared_source = _prepare_python_package_source(
            instrumented_source,
            required_modules=required_modules,
        )
        (pkg_dir / "atoms.py").write_text(prepared_source)

        # pipeline.py
        entrypoints, default_entrypoint = _discover_python_entrypoints(prepared_source)
        pipeline_content = generate_pipeline_py(
            [],
            entrypoint_names=entrypoints,
            default_entrypoint=default_entrypoint,
        )
        (pkg_dir / "pipeline.py").write_text(pipeline_content)

        # runner.py
        runner_path = output_dir / "runner.py"
        runner_path.write_text(generate_runner_py(package_name))

        # Run mypy --strict on the package
        mypy_bin = getattr(self._config, "python_mypy_path", "mypy")
        try:
            pkg_arg = str(pkg_dir.resolve())
            proc = await asyncio.create_subprocess_exec(
                mypy_bin,
                "--strict",
                "--disable-error-code",
                "import-untyped",
                "--disable-error-code",
                "valid-type",
                "--disable-error-code",
                "name-defined",
                "--disable-error-code",
                "no-any-return",
                "--disable-error-code",
                "no-untyped-def",
                "--disable-error-code",
                "no-untyped-call",
                "--disable-error-code",
                "unused-ignore",
                "--disable-error-code",
                "type-arg",
                pkg_arg,
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raw = stdout.decode() + stderr.decode()
                errors.append(
                    f"mypy validation failed (exit {proc.returncode}): {raw[:500]}"
                )
        except FileNotFoundError:
            errors.append(f"mypy not found at '{mypy_bin}' — skipping validation")

        artifact = runner_path if runner_path.exists() else None
        return artifact, runner_path if runner_path.exists() else None, errors

    def _generate_rust_ffi(
        self,
        skeleton: SkeletonFile,
        output_dir: Path,
    ) -> list[Path]:
        """Generate Rust FFI crate files."""
        ffi_dir = output_dir / "ffi"
        src_dir = ffi_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        name = skeleton.metadata.get("name", "verified")
        files: list[Path] = []

        # Cargo.toml
        cargo_path = ffi_dir / "Cargo.toml"
        cargo_path.write_text(generate_cargo_toml(name))
        files.append(cargo_path)

        # build.rs
        build_rs_path = ffi_dir / "build.rs"
        lean_lib_path = str(output_dir / ".lake" / "build" / "lib")
        build_rs_path.write_text(generate_build_rs(lean_lib_path))
        files.append(build_rs_path)

        # src/lib.rs
        lib_rs_path = src_dir / "lib.rs"
        lib_rs_path.write_text(generate_lib_rs(skeleton.units))
        files.append(lib_rs_path)

        return files

    def _generate_c_header(
        self,
        skeleton: SkeletonFile,
        output_dir: Path,
    ) -> list[Path]:
        """Generate a C header file for FFI."""
        ffi_dir = output_dir / "ffi"
        ffi_dir.mkdir(parents=True, exist_ok=True)

        header_lines: list[str] = [
            "/* Auto-generated by AGEO-Matcher extractor. */",
            "#ifndef AGEO_EXPORT_H",
            "#define AGEO_EXPORT_H",
            "",
            "#include <stdint.h>",
            "",
            "/* Opaque handle to a Lean/Coq managed object. */",
            "typedef struct lean_object lean_object;",
            "",
        ]

        for unit in skeleton.units:
            c_name = unit.declaration_name.replace(".", "_")
            header_lines.append(f"/* Verified: {unit.name} */")
            header_lines.append(f"/* Type: {unit.type_signature} */")
            header_lines.append(f"lean_object* {c_name}(lean_object* arg);")
            header_lines.append("")

        header_lines.append("#endif /* AGEO_EXPORT_H */")
        header_lines.append("")

        header_path = ffi_dir / "export.h"
        header_path.write_text("\n".join(header_lines))

        return [header_path]
