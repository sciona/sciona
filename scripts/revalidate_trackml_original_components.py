"""Reexecute retained public-source comparisons before original CDG promotion."""
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile

from scripts import audit_decomposed_trackml_readiness as baseline
from scripts.audit_trackml_cell_catalog_readiness import review as catalog_review
from scripts.promote_trackml_cell_adapter import qualify_plan, check
from scripts.validate_decomposed_trackml_cell_graph import validate as validate_graph


COMPONENTS = (
    'circle', 'helix', 'nearest', 'detector', 'intersections', 'corrections',
    'layer_integration', 'neighbors', 'bayesian', 'candidates', 'extension',
    'runtime', 'seeding', 'cells', 'lifecycle',
)


def validate():
    if not __debug__:
        raise RuntimeError('Source comparisons require Python assertions enabled')
    root = baseline.ROOT
    before = baseline.audit()
    if before['implementation_drift']:
        raise ValueError('Retained source implementation changed')
    pins_path = root/'docs/reviews/competition_trackml_source_pins.json'
    pins = json.loads(pins_path.read_text())
    vendor = root.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking'
    results = {}
    with tempfile.TemporaryDirectory(prefix='trackml-public-source-') as directory:
        source = Path(directory)
        for name in ('geometry.py', 'neighbors.py', 'corrections.py', 'candidates.py', 'cells.py', 'algorithm.py'):
            relative = 'trackml_solution/'+name
            data = (vendor/('solution_'+relative.replace('/', '_')+'.source')).read_bytes()
            if hashlib.sha256(data).hexdigest() != pins['files'][relative]:
                raise ValueError('Pinned public software changed')
            target = source/relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
        for name in COMPONENTS:
            report = root/f'docs/reviews/competition_trackml_{name}.json'
            expected = json.loads(report.read_text())
            module = importlib.import_module('scripts.validate_trackml_'+name)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                actual = module.validate(root, source)
            # JSON normalization accounts for tuples without weakening numeric checks.
            if json.loads(json.dumps(actual)) != expected:
                raise ValueError('Source comparison evidence differs: '+name)
            results[name] = dict(exact_report_match=True, report_sha256=baseline.sha(report),
                validator_sha256=baseline.sha(root/f'scripts/validate_trackml_{name}.py'))
            print('Revalidated '+name, flush=True)
    graph = validate_graph()
    stored_graph = root/'docs/reviews/decomposed_trackml_cell_graph_execution.json'
    if json.loads(json.dumps(graph)) != json.loads(stored_graph.read_text()):
        raise ValueError('Decomposed graph evidence differs')
    readiness = catalog_review()
    if readiness['unresolved_nodes'] or readiness['ready_nodes'] != 28:
        raise ValueError('All twenty-eight graph bindings must be approved and served')
    plan = qualify_plan()
    with baseline.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check(db, plan, published=True)
    if baseline.audit() != before:
        raise ValueError('Baseline changed during revalidation')
    return dict(format='trackml-original-component-revalidation.v1', passed=True,
        approved=False, catalog_mutations=0, synthetic_only=True,
        source_version_id=baseline.ORIGINAL, source_content_hash=baseline.SOURCE_HASH,
        components=results, graph_evidence_sha256=baseline.sha(stored_graph),
        graph_sha256=graph['graph_sha256'], ready_nodes=readiness['ready_nodes'],
        exact_published_adapter_dependencies=True,
        original_stage_coverage=readiness['original_stage_coverage'],
        validator_sha256=baseline.sha(__file__),
        limits=[
            'Reexecution establishes retained synthetic comparison results, not historical competition accuracy.',
            'Original scope review, corrected graph staging and original publication remain separate gates.',
            'Earlier component reports retain their historical next-work statements; they are not current completion claims.',
            'Numerical provider coverage does not alone prove complete source-stage semantics.',
            'No learning of calibration maps is certified; synthetic supplied maps exercise application.'])


if __name__ == '__main__':
    result = validate()
    target = baseline.ROOT/'docs/reviews/trackml_original_component_revalidation.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, components=len(result['components']),
        ready_nodes=result['ready_nodes'], approved=False)))
