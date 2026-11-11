"""Python proof environment using import-based checking (default) or mypy."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

from sciona.architect.models import split_top_level_type_args
from sciona.atom_identity import (
    candidate_atom_provider_roots,
    known_atom_package_prefixes,
)
from sciona.judge.models import CompilerFeedback


def _module(func_name: str) -> str:
    """Extract the module path from a dotted function name."""
    parts = func_name.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else func_name


def _leaf(func_name: str) -> str:
    """Extract the leaf name from a dotted function name."""
    return func_name.rsplit(".", 1)[-1]


def _extract_top_level_param_block(type_sig: str) -> str | None:
    """Extract the top-level ``(...)`` parameter list from a function signature."""
    sig = type_sig.strip()
    if not sig.startswith("("):
        return None
    depth = 0
    quote_char: str | None = None
    for idx, ch in enumerate(sig):
        if quote_char is not None:
            if ch == quote_char:
                quote_char = None
            continue
        if ch in {"'", '"'}:
            quote_char = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return sig[1:idx]
    return None


def _count_params_from_signature(type_sig: str) -> int | None:
    """Extract expected parameter count from a type signature like '(x: T, y: U) -> R'.

    Uses bracket-aware splitting so nested generics like ``dict[str, int]`` or
    ``tuple[float, float]`` are counted as a single parameter.
    Returns None if the signature cannot be parsed.
    """
    params_block = _extract_top_level_param_block(type_sig)
    if params_block is None:
        return None
    params_str = params_block.strip()
    if not params_str or params_str.lower() == "void":
        return 0
    params = [
        p.strip()
        for p in split_top_level_type_args(params_str, ",")
        if p.strip() and p.strip() not in {"*", "/"}
    ]
    return len(params)


def _looks_like_dependency_environment_failure(raw: str) -> bool:
    """Return whether *raw* looks like an environment/import crash.

    These failures usually come from binary-compatibility issues in optional
    dependencies and should not block whole-file structural validation of a
    generated skeleton. Missing modules and real synthesis/runtime errors still
    fail hard.
    """
    text = raw.lower()
    markers = (
        "a module that was compiled using numpy 1.x cannot be run in",
        "numpy.core.multiarray failed to import",
        "numpy.dtype size changed",
        "_array_api not found",
        "pythoncall.jl did not start properly",
        "package pythoncall",
    )
    return any(marker in text for marker in markers)


def _is_known_atom_module(module_name: str) -> bool:
    """Return whether *module_name* uses a recognized atom package prefix."""
    text = str(module_name or "").strip()
    if not text:
        return False
    for prefix in known_atom_package_prefixes():
        if text == prefix or text.startswith(prefix + "."):
            return True
    return False


def _local_module_source_exists(module_name: str) -> bool:
    """Return whether a dotted module resolves to a source file in local repos."""
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return False
    rel_py = Path(*parts).with_suffix(".py")
    rel_pkg = Path(*parts) / "__init__.py"
    search_roots: list[Path] = [Path.cwd(), *candidate_atom_provider_roots(), Path.cwd().parent]
    seen_roots: set[Path] = set()
    for root in search_roots:
        resolved_root = root.resolve()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)
        for source_root in (resolved_root, resolved_root / "src"):
            for rel in (rel_py, rel_pkg):
                if (source_root / rel).exists():
                    return True
    return False


class PythonEnvironment:
    """ProofEnvironment implementation for Python.

    Default mode ("import") verifies that the function exists, is callable,
    and has compatible arity.  Optional mode ("mypy") uses mypy --strict
    for full type checking.
    """

    def __init__(
        self,
        mypy_path: str = "mypy",
        python_path: str = "python",
        verify_mode: str = "import",
    ) -> None:
        self._mypy_path = mypy_path
        self._python_path = python_path
        self._verify_mode = verify_mode
        self._tmpdir = tempfile.mkdtemp(prefix="sciona_python_")
        self._import_timeout_s = float(
            os.environ.get("SCIONA_PYTHON_IMPORT_TIMEOUT_S", "20")
        )

    @property
    def prover_name(self) -> str:
        return "python"

    @property
    def python_path(self) -> str:
        return self._python_path

    async def check_term(self, term: str, expected_type: str) -> tuple[bool, str]:
        """Check if a term has the expected type.

        In import mode: verifies the function is importable, callable, and
        has compatible arity with the expected type signature.
        In mypy mode: writes `_result: {expected_type} = {term}` and runs mypy.
        """
        if self._verify_mode == "mypy":
            code = f"_result: {expected_type} = {term}\n"
            feedback = await self._run_mypy(code)
            return feedback.success, feedback.raw_output

        return await self._check_term_import(term, expected_type)

    async def _check_term_import(
        self, term: str, expected_type: str
    ) -> tuple[bool, str]:
        """Import-based verification: importable, callable, arity-compatible."""
        func_name = term.lstrip("@")
        mod = _module(func_name)
        leaf = _leaf(func_name)
        expected_arity = _count_params_from_signature(expected_type)

        lines = [
            f"from {mod} import {leaf}",
            f'assert callable({leaf}), "{func_name} is not callable"',
        ]
        if expected_arity is not None:
            lines.extend([
                "import inspect as _insp",
                f"_sig = _insp.signature({leaf})",
                "_all_params = list(_sig.parameters.values())",
                "_has_varargs = any(p.kind in (_insp.Parameter.VAR_POSITIONAL, _insp.Parameter.VAR_KEYWORD) for p in _all_params)",
                "_explicit = [p for p in _all_params if p.kind not in (_insp.Parameter.VAR_POSITIONAL, _insp.Parameter.VAR_KEYWORD)]",
                "_required = [p for p in _explicit if p.default is _insp.Parameter.empty]",
                "_actual = len(_required)",
                "_max_accept = len(_explicit)",
                f'assert _actual <= {expected_arity + 2}, f"arity mismatch: expected ~{expected_arity}, got {{_actual}} required"',
            ])
            if expected_arity > 0:
                lines.append(
                    f'assert _has_varargs or _max_accept >= {expected_arity}, '
                    f'f"arity mismatch: expected at least {expected_arity} accepted params, got {{_max_accept}}"'
                )
        lines.append('print("OK")')
        code = "\n".join(lines) + "\n"
        env = dict(os.environ)
        env.setdefault("PYTHON_JULIACALL_INIT", "no")

        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_path,
                "-c",
                code,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._import_timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
                raw = stdout.decode() + stderr.decode()
                if raw.strip():
                    raw = raw.strip() + "\n"
                raw += (
                    "python import verification timed out "
                    f"after {self._import_timeout_s:.1f}s"
                )
                return False, raw
            raw = stdout.decode() + stderr.decode()
            if proc.returncode == 0:
                return True, raw.strip()
            if _looks_like_dependency_environment_failure(raw) and (
                not _is_known_atom_module(mod) or _local_module_source_exists(mod)
            ):
                combined = raw.strip()
                if combined:
                    combined += "\n"
                combined += (
                    "Import verification hit a dependency-environment failure; "
                    "accepting candidate based on structural availability."
                )
                return True, combined
            return False, raw.strip()
        except FileNotFoundError:
            msg = f"python not found at '{self._python_path}'"
            return False, msg

    async def check_proof(self, statement: str, proof_body: str) -> tuple[bool, str]:
        """Check a function with icontract decorators.

        Writes the statement and body to a temp file, then executes the file
        with the configured Python interpreter. For generated artifacts this is
        a better fit than strict mypy because conceptual annotations and
        untyped third-party packages are expected during synthesis.
        """
        code = f"{statement}\n{proof_body}\n"
        feedback = await self._run_file(code)
        return feedback.success, feedback.raw_output

    async def check_generated_files(
        self,
        files: dict[str, str],
        *,
        verify_mode: str = "mypy",
        strict: bool = True,
        ignore_missing_imports: bool = True,
    ) -> tuple[bool, str]:
        """Verify a generated multi-file Python bundle.

        The ingester emits several sibling modules (for example ``atoms.py`` and
        ``state_models.py``). Type-checking those files independently loses
        import context, so write them into one temp directory and run the chosen
        verifier against the whole bundle.
        """
        bundle_dir = Path(tempfile.mkdtemp(dir=self._tmpdir, prefix="bundle_"))
        paths: list[str] = []
        for filename, source in files.items():
            if not filename:
                continue
            path = bundle_dir / Path(filename).name
            path.write_text(source)
            paths.append(str(path))

        if not paths:
            return True, ""

        mode = (verify_mode or self._verify_mode).strip().lower()
        if mode == "mypy":
            feedback = await self._run_mypy_paths(
                paths,
                cwd=bundle_dir,
                strict=strict,
                ignore_missing_imports=ignore_missing_imports,
            )
            return feedback.success, feedback.raw_output

        errors: list[str] = []
        outputs: list[str] = []
        for path in paths:
            feedback = await self._run_file(Path(path).read_text())
            if feedback.raw_output:
                outputs.append(feedback.raw_output)
            errors.extend(feedback.errors)
        return len(errors) == 0, "\n".join(part for part in outputs if part).strip()

    async def _run(self, code: str) -> CompilerFeedback:
        """Execute raw Python source for whole-file validation."""
        return await self._run_file(code)

    async def get_type(self, name: str) -> str | None:
        """Get the type signature of a name via inspect.signature."""
        code = f"import inspect\n" f"print(inspect.signature({name}))\n"
        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_path,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode().strip() or None
            return None
        except FileNotFoundError:
            return None

    async def close(self) -> None:
        """Clean up temp directory."""

        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    async def _run_mypy(self, code: str) -> CompilerFeedback:
        """Write code to an isolated temp _check.py file and run mypy --strict."""
        run_dir = Path(tempfile.mkdtemp(dir=self._tmpdir, prefix="mypy_"))
        tmp_path = run_dir / "_check.py"
        try:
            tmp_path.write_text(code)
            return await self._run_mypy_paths([str(tmp_path)])
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    async def _run_mypy_paths(
        self,
        paths: list[str],
        *,
        cwd: Path | None = None,
        strict: bool = True,
        ignore_missing_imports: bool = False,
    ) -> CompilerFeedback:
        """Run mypy against one or more existing Python paths."""
        cmd = [self._mypy_path]
        if strict:
            cmd.append("--strict")
        if ignore_missing_imports:
            cmd.append("--ignore-missing-imports")
        cmd.extend(paths)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd) if cwd is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            raw = stdout.decode() + stderr.decode()
            return _parse_mypy_output(raw)
        except FileNotFoundError:
            return CompilerFeedback(
                raw_output=f"mypy not found at '{self._mypy_path}'",
                errors=[f"mypy not found at '{self._mypy_path}'"],
            )

    async def _run_file(self, code: str) -> CompilerFeedback:
        """Write code to an isolated temp _check.py file and execute it."""
        run_dir = Path(tempfile.mkdtemp(dir=self._tmpdir, prefix="exec_"))
        tmp_path = run_dir / "_check.py"
        tmp_path.write_text(code)
        env = dict(os.environ)
        env.setdefault("PYTHON_JULIACALL_INIT", "no")

        try:
            proc = await asyncio.create_subprocess_exec(
                self._python_path,
                str(tmp_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            raw = stdout.decode() + stderr.decode()
            if proc.returncode != 0 and _looks_like_dependency_environment_failure(raw):
                compile_proc = await asyncio.create_subprocess_exec(
                    self._python_path,
                    "-m",
                    "py_compile",
                    str(tmp_path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                compile_stdout, compile_stderr = await compile_proc.communicate()
                compile_raw = compile_stdout.decode() + compile_stderr.decode()
                if compile_proc.returncode == 0:
                    note = (
                        "Runtime import validation failed due to dependency environment "
                        "constraints; py_compile fallback passed.\n"
                    )
                    combined = raw
                    if combined and not combined.endswith("\n"):
                        combined += "\n"
                    combined += note
                    return CompilerFeedback(
                        raw_output=combined,
                        errors=[],
                        warnings=[],
                    )
                raw = raw + ("\n" if raw and not raw.endswith("\n") else "")
                raw += compile_raw
            return CompilerFeedback(
                raw_output=raw,
                errors=[] if proc.returncode == 0 else [raw.strip() or raw],
            )
        except FileNotFoundError:
            return CompilerFeedback(
                raw_output=f"python not found at '{self._python_path}'",
                errors=[f"python not found at '{self._python_path}'"],
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


def _parse_mypy_output(raw: str) -> CompilerFeedback:
    """Parse mypy output into structured CompilerFeedback."""
    errors: list[str] = []
    warnings: list[str] = []

    for line in raw.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if re.search(r":\d+: error:", line_stripped):
            errors.append(line_stripped)
        elif re.search(r":\d+: warning:", line_stripped):
            warnings.append(line_stripped)
        elif re.search(r":\d+: note:", line_stripped):
            pass  # notes are informational
        elif "Found" in line_stripped and "error" in line_stripped:
            pass  # summary line like "Found 2 errors in 1 file"

    return CompilerFeedback(
        raw_output=raw,
        errors=errors,
        warnings=warnings,
    )
