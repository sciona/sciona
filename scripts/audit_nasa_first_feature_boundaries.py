"""Synthetic counterexamples for the pinned first-place feature boundary."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def audit(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned software changed')
    tree=ast.parse(data)
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    entry=funcs['catboost_predictions']
    calls=[n.func.id for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    extractors=[name for name in calls if name.startswith('extract_')]
    expected=['extract_mfs_features','extract_config_features','extract_etd_features','extract_moment_features','extract_weather_features','extract_runway_features','extract_standtime_features']
    if extractors!=expected:raise ValueError('Feature orchestration changed')
    adjust=funcs['adjust_master']
    namespace={'pd':pd,'np':np,'OBJECT_COLS':[]}
    # The configured categorical exemption list is explicitly injected empty:
    # these cases isolate a synthetic numeric column, not categorical processing.
    exec(compile(ast.Module(body=[adjust],type_ignores=[]),'pinned_numeric_boundary','exec'),namespace)
    run=namespace['adjust_master']
    field='synthetic_numeric_feature'
    alone=run(pd.DataFrame({field:[np.nan]}))[field].to_numpy()
    appended=run(pd.DataFrame({field:[np.nan,91.]}))[field].to_numpy()
    preceding=run(pd.DataFrame({field:[37.,np.nan]}))[field].to_numpy()
    overflow=run(pd.DataFrame({field:[40000.,-40000.,1.75,-1.75]}))[field].to_numpy()
    np.testing.assert_array_equal(alone,[0])
    np.testing.assert_array_equal(appended,[91,91])
    np.testing.assert_array_equal(preceding,[37,37])
    np.testing.assert_array_equal(overflow,[-25536,25536,1,-1])
    # Detect whether differencing occurs after a group boundary in the source.
    shifts=[n for n in ast.walk(funcs['extract_etd_features']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='shift']
    ungrouped=sum(isinstance(n.func.value,ast.Subscript) and isinstance(n.func.value.value,ast.Name) for n in shifts)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,feature_extractors=extractors,
        feature_function_sha256={name:hashlib.sha256(ast.dump(funcs[name],include_attributes=False).encode()).hexdigest() for name in expected+['adjust_master']},
        source_sha256=PIN,numeric_boundary_counterexamples=4,ungrouped_estimate_shift_calls=ungrouped,
        findings=['A missing numeric query alone receives zero; adding another query changes it to that query value through whole-table backward filling.',
            'Forward filling also crosses row/entity boundaries because adjustment has no grouping contract.',
            'Out-of-range numeric values silently wrap during signed int16 conversion; fractional values truncate toward zero.',
            'Seven feature families precede this boundary; similarly named third-place adapters are not interchangeable without source comparison.',
            'Estimated-time feature code includes ungrouped shifts; boundary behavior needs isolated runtime validation before reuse.'],
        required_gates=['Explicit per-feature missing-value policy independent of unrelated inference queries, qualified consistently for training and inference.',
            'Explicit truncation/range contract; reject unsupported values or document an independently reviewed wider representation.',
            'Complete source comparison for all seven feature families, including time availability, grouped differences, categorical mappings and joins.',
            'Training/inference corrected feature compatibility with all 21 fitted-model slots and fallback routing.'],
        limitations=['Numeric-boundary execution uses an explicitly empty categorical exemption list; no full categorical-config or raw-feature runtime claim.',
            'Ungrouped-shift finding is static evidence, not yet a demonstrated numerical defect.',
            'No corrected feature adapter, CDG registration or approval applied.'],
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_feature_boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,feature_families=len(report['feature_extractors']),numeric_counterexamples=report['numeric_boundary_counterexamples'],approved=False)))
