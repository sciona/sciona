"""Re-execute the numerical graph against pinned provider/framework dependencies."""
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import argparse

from scripts.validate_residual_classifier_graph import validate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(source):
    review_path = ROOT/'docs/reviews/residual_classifier_binding_review.json'
    dependencies_path = ROOT/'docs/reviews/residual_classifier_provider_dependencies.json'
    review = json.loads(review_path.read_text())
    dependencies = json.loads(dependencies_path.read_text())
    if not review['passed'] or not dependencies['passed'] or dependencies['binding_review_sha256'] != sha(review_path):
        raise ValueError('Matching passing dependency and binding reviews required')
    paths = {name: Path(importlib.import_module(name).__file__).resolve()
             for name in dependencies['provider_source_sha256']}
    for name, expected in dependencies['provider_source_sha256'].items():
        if sha(paths[name]) != expected:
            raise ValueError('Provider dependency differs: '+name)
    for name in ['sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py',
        'sciona/ghost/abstract.py', 'sciona/ghost/registry.py', 'sciona/residual_classifier_graph.py',
        'sciona/nasa_workflow_inputs.py', 'sciona/nasa_feature_adapters.py', 'sciona/tabular_contracts.py',
        'scripts/validate_residual_classifier_graph.py', 'scripts/validate_nasa_corrected_workflow.py',
        'scripts/validate_nasa_third_training_sequence.py']:
        paths[name] = ROOT/name
    # Domain history provider is exercised by preparation, outside the numerical graph.
    name = 'sciona.atoms.ml.calibration.adaptive_history'
    paths[name] = Path(importlib.import_module(name).__file__).resolve()
    pins = {name:sha(path) for name,path in paths.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        execution = validate(source)
    if pins != {name:sha(path) for name,path in paths.items()}:
        raise ValueError('Execution source changed during qualification')
    if not execution['passed'] or execution['graph_sha256'] != review['graph_sha256']:
        raise ValueError('Qualified graph differs from review')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        graph_sha256=execution['graph_sha256'], binding_review_sha256=sha(review_path),
        dependency_review_sha256=sha(dependencies_path), execution_source_sha256=pins,
        scenarios=execution['scenarios'], nodes=execution['nodes'], edges=execution['edges'],
        qualifier_sha256=sha(Path(__file__)),
        limitations=['Provisioned in-process runtime only; model intermediates were kept in memory.',
            'Explicit framework, validation, adapter and transitive provider source pins checked before and after execution.',
            'Source-level adapter integration is exercised, but adapters are not catalog-facing graph nodes yet.',
            'Remaining license review and transactional publication gates are not implied by this result.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    args=parser.parse_args()
    report=qualify(args.source_directory)
    (ROOT/'docs/reviews/residual_classifier_bound_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],source_files_pinned=len(report['execution_source_sha256']),
        scenarios=len(report['scenarios']),predictions=sum(s['predictions'] for s in report['scenarios']),catalog_mutations=0)))
