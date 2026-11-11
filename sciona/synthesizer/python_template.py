"""Python package template generation for verified skeletons."""

from __future__ import annotations

import json

from sciona.synthesizer.models import SkeletonFile


def generate_pyproject_toml(package_name: str, dependencies: list[str]) -> str:
    """Generate a pyproject.toml for the output package."""
    deps = "\n".join(f'    "{dep}",' for dep in dependencies)
    return f"""\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{package_name}"
version = "0.1.0"
description = "AGEO-Matcher verified Python package"
requires-python = ">=3.10"
dependencies = [
    "icontract>=2.6",
{deps}
]

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
"""


def generate_init_py(package_name: str, exports: list[str]) -> str:
    """Generate an __init__.py with explicit exports."""
    export_lines = "\n".join(f'    "{name}",' for name in exports)
    import_lines = "\n".join(
        f"from {package_name}.atoms import {name}" for name in exports
    )
    return f"""\
\"""{package_name}: AGEO-Matcher verified package.\"""

{import_lines}

__all__ = [
{export_lines}
]
"""


def generate_main_script(
    skeleton: SkeletonFile,
    wrappers: list[str],
    pipeline_steps: list[str],
) -> str:
    """Generate atoms.py with icontract-wrapped safe atoms."""
    lines: list[str] = []
    lines.append('"""Safe atom wrappers with icontract contracts."""')
    lines.append("")
    lines.append("import icontract")
    lines.append("")

    # Collect unique imports from skeleton metadata
    imports_seen: set[str] = set()
    for unit in skeleton.units:
        module = (
            unit.declaration_name.rsplit(".", 1)[0]
            if "." in unit.declaration_name
            else ""
        )
        if module and module not in imports_seen:
            imports_seen.add(module)
            lines.append(f"import {module}")

    if imports_seen:
        lines.append("")

    # Wrapper functions
    for wrapper_code in wrappers:
        lines.append("")
        lines.append(wrapper_code)
        lines.append("")

    return "\n".join(lines)


def generate_pipeline_py(
    pipeline_steps: list[str],
    *,
    entrypoint_names: list[str] | None = None,
    default_entrypoint: str | None = None,
) -> str:
    """Generate pipeline.py that wires atoms per CDG topology."""
    entrypoint_names = entrypoint_names or []
    entrypoints_json = json.dumps(entrypoint_names)
    default_json = json.dumps(default_entrypoint)
    lines: list[str] = []
    lines.append("# mypy: disable-error-code=no-any-return")
    lines.append('"""Pipeline orchestration: wires atoms per CDG topology."""')
    lines.append("")
    lines.append("import argparse")
    lines.append("import inspect")
    lines.append("import json")
    lines.append("import os")
    lines.append("from pathlib import Path")
    lines.append("from . import atoms")
    lines.append("from typing import Any")
    lines.append("try:")
    lines.append("    from sciona.principal.datasets import create_templated_dataset_collection")
    lines.append("    from sciona.principal.datasets._parser import DataSetCollection")
    lines.append("    from sciona.principal.dataset_slice import apply_relative_dataset_slice")
    lines.append("    from sciona.principal.eval_spec import compute_evaluation_payload, load_evaluation_spec")
    lines.append("    from sciona.principal.datasets.io import read as read_dataset_template")
    lines.append("    from sciona.principal.runtime_context import canonicalize_runtime_inputs")
    lines.append("    _HAS_SCIONA_PRINCIPAL = True")
    lines.append("except ImportError:")
    lines.append("    _HAS_SCIONA_PRINCIPAL = False")
    lines.append("    def canonicalize_runtime_inputs(flat: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:")
    lines.append("        return dict(flat), {}")
    lines.append("    def load_evaluation_spec(spec_arg: str | None) -> dict[str, Any] | None:")
    lines.append("        if not spec_arg:")
    lines.append("            return None")
    lines.append("        p = Path(spec_arg)")
    lines.append("        return json.loads(p.read_text()) if p.exists() else json.loads(spec_arg)")
    lines.append("    def compute_evaluation_payload(result: Any, flat_inputs: dict[str, Any], eval_spec: dict[str, Any]) -> dict[str, Any]:")
    lines.append("        raise RuntimeError('Evaluation spec execution requires sciona.principal.eval_spec')")
    lines.append("")
    lines.append(f"ENTRYPOINTS = {entrypoints_json}")
    lines.append(f"DEFAULT_ENTRYPOINT = {default_json}")
    lines.append("SIGNAL_GROUP_TOKENS = ('signal', 'wave', 'waveform', 'ecg', 'ppg', 'eeg', 'emg')")
    lines.append("")
    lines.append("")
    lines.append("def _parse_dataset_vars(values: list[str] | None) -> dict[str, str]:")
    lines.append("    result: dict[str, str] = {}")
    lines.append("    for item in values or []:")
    lines.append("        if '=' not in item:")
    lines.append("            raise ValueError(f'dataset var must be KEY=VALUE, got: {item!r}')")
    lines.append("        key, value = item.split('=', 1)")
    lines.append("        result[key] = value")
    lines.append("    return result")
    lines.append("")
    lines.append("def _jsonify(value: Any) -> Any:")
    lines.append("    if hasattr(value, 'tolist'):")
    lines.append("        return value.tolist()")
    lines.append("    if isinstance(value, dict):")
    lines.append("        return {str(k): _jsonify(v) for k, v in value.items()}")
    lines.append("    if isinstance(value, (list, tuple)):")
    lines.append("        return [_jsonify(v) for v in value]")
    lines.append("    return value")
    lines.append("")
    lines.append("def _jsonify_compact(value: Any) -> Any:")
    lines.append("    if hasattr(value, 'shape') and hasattr(value, 'reshape') and hasattr(value, 'tolist'):")
    lines.append("        flat = value.reshape(-1)")
    lines.append("        head = flat[:8].tolist()")
    lines.append("        return {")
    lines.append("            'kind': 'ndarray',")
    lines.append("            'shape': list(value.shape),")
    lines.append("            'size': int(getattr(value, 'size', len(head))),")
    lines.append("            'head': _jsonify(head),")
    lines.append("        }")
    lines.append("    if isinstance(value, dict):")
    lines.append("        items = list(value.items())")
    lines.append("        compact = {str(k): _jsonify_compact(v) for k, v in items[:8]}")
    lines.append("        if len(items) > 8:")
    lines.append("            compact['_truncated_keys'] = len(items) - 8")
    lines.append("        return compact")
    lines.append("    if isinstance(value, (list, tuple)):")
    lines.append("        head = [_jsonify_compact(v) for v in value[:8]]")
    lines.append("        if len(value) > 8:")
    lines.append("            return {'kind': type(value).__name__, 'length': len(value), 'head': head}")
    lines.append("        return head")
    lines.append("    return _jsonify(value)")
    lines.append("")
    lines.append("def _resolve_adapter(dataset_root: str | Path) -> Path:")
    lines.append("    root = Path(dataset_root).expanduser().resolve()")
    lines.append("    candidates = ('sciona.yml', 'ageom.yml', 'adapter.yml')")
    lines.append("    for filename in candidates:")
    lines.append("        adapter = root / filename")
    lines.append("        if adapter.exists():")
    lines.append("            return adapter")
    lines.append("    names = ', '.join(candidates)")
    lines.append("    raise FileNotFoundError(f'No adapter file found in dataset root: {root} (looked for {names})')")
    lines.append("")
    lines.append("def _group_alias(group_name: str) -> str | None:")
    lines.append("    parts = [part for part in str(group_name).split('_') if part]")
    lines.append("    if not parts:")
    lines.append("        return None")
    lines.append("    alias = parts[-1]")
    lines.append("    return alias or None")
    lines.append("")
    lines.append("def _group_output_names(group_name: str, group_spec: dict[str, Any], *, include_sampling: bool = True) -> set[str]:")
    lines.append("    names = {str(group_name)}")
    lines.append("    group_lower = str(group_name).lower()")
    lines.append("    group_alias = _group_alias(str(group_name))")
    lines.append("    if include_sampling:")
    lines.append("        names.add('sampling_rate')")
    lines.append("        names.add(f'{group_name}_sampling_rate')")
    lines.append("        if group_alias:")
    lines.append("            names.add(f'{group_alias}_sampling_rate')")
    lines.append("    properties = group_spec.get('properties', {}) if isinstance(group_spec, dict) else {}")
    lines.append("    prefix = f'{group_name}_'")
    lines.append("    for prop_name in properties:")
    lines.append("        prop = str(prop_name)")
    lines.append("        names.add(f'{prefix}{prop}')")
    lines.append("        names.add(prop)")
    lines.append("        if group_alias and prop != 'value':")
    lines.append("            names.add(f'{group_alias}_{prop}')")
    lines.append("        if prop == 'value' and group_alias:")
    lines.append("            names.add(group_alias)")
    lines.append("            if any(token in group_lower for token in SIGNAL_GROUP_TOKENS):")
    lines.append("                names.add('signal')")
    lines.append("    return names")
    lines.append("")
    lines.append("def _reduce_adapter_template(")
    lines.append("    template: dict[str, Any],")
    lines.append("    entrypoint: str | None,")
    lines.append("    eval_spec: dict[str, Any] | None = None,")
    lines.append(") -> tuple[dict[str, Any], tuple[str, ...]]:")
    lines.append("    groups = template.get('groups', {}) if isinstance(template, dict) else {}")
    lines.append("    if not entrypoint or not groups:")
    lines.append("        return template, tuple(groups)")
    lines.append("    fn = getattr(atoms, entrypoint)")
    lines.append("    params = {")
    lines.append("        name")
    lines.append("        for name, param in inspect.signature(fn).parameters.items()")
    lines.append("        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)")
    lines.append("    }")
    lines.append("    if not params:")
    lines.append("        return template, tuple(groups)")
    lines.append("    content_params = {")
    lines.append("        name for name in params")
    lines.append("        if name != 'sampling_rate' and not name.endswith('_sampling_rate')")
    lines.append("    }")
    lines.append("    extra_sources = set()")
    lines.append("    if isinstance(eval_spec, dict):")
    lines.append("        prediction = eval_spec.get('prediction', {})")
    lines.append("        reference = eval_spec.get('reference', {})")
    lines.append("        for key in ('value_source', 'time_source'):")
    lines.append("            value = reference.get(key)")
    lines.append("            if isinstance(value, str) and value:")
    lines.append("                extra_sources.add(value)")
    lines.append("        if str(prediction.get('time_kind', '')).lower() == 'index':")
    lines.append("            value = prediction.get('time_source')")
    lines.append("            if isinstance(value, str) and value:")
    lines.append("                extra_sources.add(value)")
    lines.append("    selected = []")
    lines.append("    if extra_sources:")
    lines.append("        selected = [")
    lines.append("            name")
    lines.append("            for name, spec in groups.items()")
    lines.append("            if extra_sources & _group_output_names(str(name), spec, include_sampling=True)")
    lines.append("        ]")
    lines.append("    if not selected:")
    lines.append("        selected = [")
    lines.append("            name")
    lines.append("            for name, spec in groups.items()")
    lines.append("            if content_params & _group_output_names(str(name), spec, include_sampling=False)")
    lines.append("        ]")
    lines.append("    if not selected:")
    lines.append("        selected = [")
    lines.append("            name")
    lines.append("            for name, spec in groups.items()")
    lines.append("            if params & _group_output_names(str(name), spec, include_sampling=True)")
    lines.append("        ]")
    lines.append("    if extra_sources:")
    lines.append("        for name, spec in groups.items():")
    lines.append("            if name in selected:")
    lines.append("                continue")
    lines.append("            if extra_sources & _group_output_names(str(name), spec, include_sampling=True):")
    lines.append("                selected.append(name)")
    lines.append("    unresolved_params = set(content_params)")
    lines.append("    for name in selected:")
    lines.append("        spec = groups.get(name, {})")
    lines.append("        unresolved_params -= _group_output_names(str(name), spec, include_sampling=False)")
    lines.append("    if unresolved_params:")
    lines.append("        for name, spec in groups.items():")
    lines.append("            if name in selected:")
    lines.append("                continue")
    lines.append("            outputs = _group_output_names(str(name), spec, include_sampling=False)")
    lines.append("            if unresolved_params & outputs:")
    lines.append("                selected.append(name)")
    lines.append("                unresolved_params -= outputs")
    lines.append("            if not unresolved_params:")
    lines.append("                break")
    lines.append("    if not selected:")
    lines.append("        return template, tuple(groups)")
    lines.append("    reduced = dict(template)")
    lines.append("    reduced['groups'] = {name: groups[name] for name in selected}")
    lines.append("    return reduced, tuple(selected)")
    lines.append("")
    lines.append("def _materialize_adapter_template(template: dict[str, Any], selected_groups: tuple[str, ...]) -> Path:")
    lines.append("    cache_dir = Path(__file__).resolve().parents[2] / '.sciona_generated_adapters'")
    lines.append("    cache_dir.mkdir(parents=True, exist_ok=True)")
    lines.append("    target = cache_dir / ('.sciona_' + '_'.join(selected_groups or ('all',)) + '_runner.yml')")
    lines.append("    target.write_text(json.dumps(template, indent=2) + '\\n')")
    lines.append("    return target")
    lines.append("")
    lines.append("def load_dataset(")
    lines.append("    dataset_root: str | Path,")
    lines.append("    dataset_vars: dict[str, str] | None = None,")
    lines.append("    user: str | None = None,")
    lines.append("    serial: str | None = None,")
    lines.append("    entrypoint: str | None = None,")
    lines.append("    eval_spec: dict[str, Any] | None = None,")
    lines.append("    slice_start: float | None = None,")
    lines.append("    slice_stop: float | None = None,")
    lines.append(") -> dict[str, Any]:")
    lines.append("    adapter = _resolve_adapter(dataset_root)")
    lines.append("    template = read_dataset_template(str(adapter.parent), adapter.stem, varset=dataset_vars)")
    lines.append("    template, selected_groups = _reduce_adapter_template(template, entrypoint, eval_spec)")
    lines.append("    template_key = _materialize_adapter_template(template, selected_groups)")
    lines.append("    coll_cls = create_templated_dataset_collection(str(template_key), template=template)")
    lines.append("    options = coll_cls.get_filter_options(user, serial, recursive=True)")
    lines.append("    coll = DataSetCollection.from_folder(adapter.parent, subcls=coll_cls, options=options)")
    lines.append("    if slice_start is not None or slice_stop is not None:")
    lines.append("        apply_relative_dataset_slice(coll, start_s=slice_start, stop_s=slice_stop)")
    lines.append("        if getattr(coll, 'data', None) is not None:")
    lines.append("            return dict(coll.data.to_pandas(private=False, exclude=('*_start', '*_stop')))")
    lines.append("        return {}")
    lines.append("    return dict(coll.to_pandas())")
    lines.append("")
    lines.append("def _infer_sampling_rate(frame: Any) -> float | None:")
    lines.append("    if not hasattr(frame, 'columns'):")
    lines.append("        return None")
    lines.append("    columns = [str(column) for column in frame.columns]")
    lines.append("    time_column = None")
    lines.append("    if 't' in columns:")
    lines.append("        time_column = 't'")
    lines.append("    else:")
    lines.append("        for column in columns:")
    lines.append("            if column.endswith('_t'):")
    lines.append("                time_column = column")
    lines.append("                break")
    lines.append("    if time_column is None:")
    lines.append("        return None")
    lines.append("    times = frame[time_column]")
    lines.append("    values = times.to_numpy() if hasattr(times, 'to_numpy') else times")
    lines.append("    if values is None or len(values) < 2:")
    lines.append("        return None")
    lines.append("    diffs: list[float] = []")
    lines.append("    prev: float | None = None")
    lines.append("    for raw in values:")
    lines.append("        try:")
    lines.append("            current = float(raw)")
    lines.append("        except (TypeError, ValueError):")
    lines.append("            prev = None")
    lines.append("            continue")
    lines.append("        if prev is not None:")
    lines.append("            delta = current - prev")
    lines.append("            if delta > 0:")
    lines.append("                diffs.append(delta)")
    lines.append("        prev = current")
    lines.append("    if not diffs:")
    lines.append("        return None")
    lines.append("    diffs.sort()")
    lines.append("    median = diffs[len(diffs) // 2]")
    lines.append("    if median <= 0:")
    lines.append("        return None")
    lines.append("    return 1.0 / median")
    lines.append("")
    lines.append("def _flatten_inputs(kwargs: dict[str, Any]) -> dict[str, Any]:")
    lines.append("    flat = {key: value for key, value in kwargs.items() if not hasattr(value, 'columns')}")
    lines.append("    for group_name, frame in list(kwargs.items()):")
    lines.append("        if not hasattr(frame, 'columns'):")
    lines.append("            continue")
    lines.append("        group_lower = str(group_name).lower()")
    lines.append("        group_alias = _group_alias(str(group_name))")
    lines.append("        prefix = f'{group_name}_'")
    lines.append("        for column in frame.columns:")
    lines.append("            series = frame[column]")
    lines.append("            value = series.to_numpy() if hasattr(series, 'to_numpy') else series")
    lines.append("            flat[str(column)] = value")
    lines.append("            alias = None")
    lines.append("            if isinstance(column, str) and column.startswith(prefix):")
    lines.append("                alias = column[len(prefix):]")
    lines.append("                flat.setdefault(alias, value)")
    lines.append("                if group_alias:")
    lines.append("                    if alias == 'value':")
    lines.append("                        flat.setdefault(group_alias, value)")
    lines.append("                    else:")
    lines.append("                        flat.setdefault(f'{group_alias}_{alias}', value)")
    lines.append("        sampling_rate = _infer_sampling_rate(frame)")
    lines.append("        if sampling_rate is not None:")
    lines.append("            flat.setdefault(f'{group_name}_sampling_rate', sampling_rate)")
    lines.append("    flat, _ = canonicalize_runtime_inputs(flat)")
    lines.append("    return flat")
    lines.append("")
    lines.append("def _select_call_kwargs(fn: Any, flat_inputs: dict[str, Any]) -> dict[str, Any]:")
    lines.append("    signature = inspect.signature(fn)")
    lines.append("    result: dict[str, Any] = {}")
    lines.append("    for name, param in signature.parameters.items():")
    lines.append("        if name in flat_inputs:")
    lines.append("            result[name] = flat_inputs[name]")
    lines.append("        elif param.default is not inspect._empty:")
    lines.append("            continue")
    lines.append("    return result")
    lines.append("")
    lines.append("def run_pipeline(**kwargs: Any) -> Any:")
    lines.append('    """Execute the verified pipeline."""')
    lines.append("    if not ENTRYPOINTS:")
    lines.append('        raise NotImplementedError("No exported entrypoints are available")')
    lines.append("    entrypoint = kwargs.pop('entrypoint', None) or DEFAULT_ENTRYPOINT or ENTRYPOINTS[0]")
    lines.append("    fn = getattr(atoms, entrypoint)")
    lines.append("    flat_inputs = _flatten_inputs(kwargs)")
    lines.append("    call_kwargs = _select_call_kwargs(fn, flat_inputs)")

    if pipeline_steps:
        for step in pipeline_steps:
            lines.append(f"    {step}")
    else:
        lines.append("    return fn(**call_kwargs)")

    lines.append("")
    lines.append("")
    lines.append("def main(argv: list[str] | None = None) -> int:")
    lines.append("    parser = argparse.ArgumentParser()")
    lines.append("    parser.add_argument('--dataset-root', required=True)")
    lines.append("    parser.add_argument('--dataset-var', action='append', default=[])")
    lines.append("    parser.add_argument('--user', default=None)")
    lines.append("    parser.add_argument('--serial', default=None)")
    lines.append("    parser.add_argument('--slice-start', type=float, default=None)")
    lines.append("    parser.add_argument('--slice-stop', type=float, default=None)")
    lines.append("    parser.add_argument('--entrypoint', default=None)")
    lines.append("    parser.add_argument('--trace-path', default='trace.jsonl')")
    lines.append("    parser.add_argument('--eval-spec', default=None)")
    lines.append("    parser.add_argument('--params', default=None)")
    lines.append("    parser.add_argument('--output', default=None)")
    lines.append("    args = parser.parse_args(argv)")
    lines.append("")
    lines.append("    dataset_vars = _parse_dataset_vars(args.dataset_var)")
    lines.append("    eval_spec = load_evaluation_spec(args.eval_spec)")
    lines.append("    if hasattr(atoms, '_SCIONA_TRACE_PATH'):")
    lines.append("        atoms._SCIONA_TRACE_PATH = args.trace_path")
    lines.append("    if args.params and hasattr(atoms, '_SCIONA_PARAMS'):")
    lines.append("        atoms._SCIONA_PARAMS = json.loads(Path(args.params).read_text())")
    lines.append("    entrypoint = args.entrypoint or DEFAULT_ENTRYPOINT or (ENTRYPOINTS[0] if ENTRYPOINTS else None)")
    lines.append("    group_frames = load_dataset(")
    lines.append("        args.dataset_root,")
    lines.append("        dataset_vars=dataset_vars or None,")
    lines.append("        user=args.user,")
    lines.append("        serial=args.serial,")
    lines.append("        slice_start=args.slice_start,")
    lines.append("        slice_stop=args.slice_stop,")
    lines.append("        entrypoint=entrypoint,")
    lines.append("        eval_spec=eval_spec,")
    lines.append("    )")
    lines.append("    flat_inputs = _flatten_inputs(group_frames)")
    lines.append("    result = run_pipeline(entrypoint=entrypoint, **group_frames)")
    lines.append("    compact_output = str(os.environ.get('SCIONA_COMPACT_OUTPUT', '')).lower() in {'1', 'true', 'yes', 'on'}")
    lines.append("    payload = {")
    lines.append("        'mse': None,")
    lines.append("        'loss': None,")
    lines.append("        'entrypoint': entrypoint,")
    lines.append("        'outputs': _jsonify_compact(result) if compact_output else _jsonify(result),")
    lines.append("    }")
    lines.append("    if eval_spec is not None:")
    lines.append("        payload.update(compute_evaluation_payload(result, flat_inputs, eval_spec))")
    lines.append("    text = json.dumps(payload)")
    lines.append("    if args.output:")
    lines.append("        Path(args.output).write_text(text + '\\n')")
    lines.append("    print(text)")
    lines.append("    return 0")
    lines.append("")

    return "\n".join(lines)


def generate_runner_py(package_name: str) -> str:
    """Generate a top-level runner that executes the exported pipeline package."""
    return f"""\
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from {package_name}.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
"""
