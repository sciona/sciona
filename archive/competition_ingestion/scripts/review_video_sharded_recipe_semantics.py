"""Qualify recipe policy and all nine exact numerical dependency bindings."""
import json
import subprocess

from scripts.freeze_video_graph_contracts import freeze, PATH, ROOT, sha
from scripts.stage_video_sharded_recipe import reviewed_plan, connect, preserved
from sciona.catalog_video_multilabel_graph import build_catalog_video_multilabel_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.architect.models import EdgeKind


def review():
    intake = reviewed_plan()
    frozen = freeze()
    if frozen != json.loads(PATH.read_text()):
        raise ValueError('Frozen catalog contracts changed')
    evidence = 'docs/reviews/catalog_video_multilabel_execution.json'
    raw = subprocess.check_output(['git', 'show', 'f93bc1ba:'+evidence], cwd=ROOT)
    if (ROOT/evidence).read_bytes() != raw:
        raise ValueError('Immutable execution evidence changed')
    execution = json.loads(raw)
    if not execution['passed'] or execution['tests'] != dict(tests=3, failures=0, errors=0, skipped=0):
        raise ValueError('Complete configured lifecycle evidence required')
    for path, digest in execution['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Executed software changed')
    graph = build_catalog_video_multilabel_graph()
    if encode_execution_graph(graph)[0] != execution['graph_sha256']:
        raise ValueError('Executed graph changed')
    records = {r['node_id']: r for r in frozen['records']}
    dependencies = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if preserved(db) != intake['preserved_sha256']:
            raise ValueError('Original or derivative history changed')
        for record in records.values():
            if record['ports'] != record['legacy_ports']:
                raise ValueError('Legacy and canonical contracts differ')
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                expected = dict(status=('flagged' if prefix == 'atom' else 'draft'), is_publishable=False) if record['node_id'] == 'execute' else dict(status='approved', is_publishable=True)
                actual = db.execute(f'SELECT status,is_publishable FROM {prefix}s WHERE {key}=%s', (record['artifact_id'],)).fetchone()
                if actual != expected:
                    raise ValueError('Provider state differs')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')", (record['version_id'],)).fetchone():
                raise ValueError('Completed failed provider audit')
        recipe = records['execute']
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (recipe['version_id'],)).fetchone():
            raise ValueError('Unexpected preexisting recipe dependencies')
        for edge in graph.edges:
            if edge.edge_kind != EdgeKind.CALLABLE_INJECTION or edge.target_id != 'execute':
                raise ValueError('Exact callable dependency edges required')
            record = records[edge.source_id]
            dependencies.append(dict(port_name=edge.input_name,
                dependency_artifact_fqdn=record['fqdn'], dependency_content_hash=record['content_hash'],
                dependency_role='logic_atom', optional=False, runtime_fqdn=record['runtime_fqdn'],
                version_id=record['version_id'], interfaces=record['ports'],
                legacy_versioned_interface_present=True))
    if len(dependencies) != 9 or len({d['port_name'] for d in dependencies}) != 9:
        raise ValueError('Nine distinct numerical dependency ports required')
    from sciona.atoms.ml.video_multilabel_sharded_execution import execute
    from sciona.ghost.registry import get_witness
    try:
        get_witness(execute.__module__+'.'+execute.__name__)()
    except ValueError:
        pass
    else:
        raise ValueError('Abstract recipe execution must reject')
    return dict(format='video-sharded-recipe-semantics.v1', passed=True, approved=False,
        proposed_tier=3, verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True,
        adapter=intake['providers'][0], registered_provider_dependencies=sorted(dependencies, key=lambda d:d['port_name']),
        execution=execution, execution_evidence_sha256=sha(ROOT/evidence),
        intake_plan_sha256=sha(ROOT/'docs/reviews/video_sharded_recipe_intake_plan.json'),
        reviewer_sha256=sha(__file__), abstract_witness_rejected=True,
        policy_review=dict(
            populations='Require labelled training/calibration and unlabelled query, aligned widths, disjoint groups and both classes per training/calibration label.',
            optimization='Full caller-configured epochs and batches; learned attention and context gates; CPU float64 Adam with binary cross entropy.',
            sparse_branch='Training-only sorted vocabulary, one L2 LBFGS classifier per label with C=1, tolerance=1e-4, max_iter=1000 and caller seed.',
            combination='Equal ordered neural/sparse mean per label; empirical held-out F1 thresholds with inclusive application to every query row.',
            reuse='Nine independently selectable numerical atoms; population, model, optimizer and split policies remain explicit in this recipe.'),
        limitations=[
            'Automated Tier 3 only; no human Tier 1 or usage-qualified Tier 2 claim.',
            'Three complete synthetic catalog-graph executions are immutable prior evidence, not rerun by this review; earlier 100-test numerical evidence is source-pinned by intake qualification.',
            'Only the nine exact reviewed callables are qualified; arbitrary substitution is not covered.',
            'Caller-owned FeatureShards stores are concrete runtime objects, not portable serialized inputs. Private data and paths are excluded from catalog artifacts.',
            'Frame reads are bounded; training labels, sparse CSR, model state and output scores remain resident. No constant-total-memory guarantee.',
            'No historical competition accuracy, rare-label improvement or generalization guarantee.',
            'Runtime qualification covers the installed CPU environment; package file integrity does not establish publisher authentication or portability.',
            'Original CDG scope and publication remain separate gates.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/video_sharded_recipe_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, dependencies=len(result['registered_provider_dependencies']), approved=False)))
