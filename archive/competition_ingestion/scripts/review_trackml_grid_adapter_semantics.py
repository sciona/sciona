"""Qualify the concrete tracking adapter and its explicit numeric dependency bindings."""
import base64
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import json
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from scripts.stage_trackml_grid_adapter import reviewed_plan, ROOT, sha, connect
from scripts.review_trackml_grid_provider_semantics import staged
from scripts.audit_trackml_grid_catalog_readiness import review as catalog_review
from scripts.validate_decomposed_trackml_grid_graph import validate as execute_comparison
from sciona.decomposed_trackml_grid_graph import build_decomposed_trackml_grid_graph
from sciona.architect.models import EdgeKind

def packages():
    pending = ['numpy','sympy','icontract','pydantic','threadpoolctl','scikit-learn','pandas']
    result = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in result: continue
        dist = metadata.distribution(name); requirements = []
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({'extra': ''}): continue
            if req.extras or req.url: raise ValueError('Unreviewed extra or URL requirement')
            if not req.specifier.contains(metadata.version(req.name), prereleases=True):
                raise ValueError('Unsatisfied runtime requirement: '+str(req))
            requirements.append(str(req)); pending.append(req.name)
        verified = 0
        for relative in dist.files or []:
            if relative.hash is None: continue
            path = dist.locate_file(relative)
            with path.open('rb') as stream: digest = hashlib.file_digest(stream, relative.hash.mode).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b'=').decode() != relative.hash.value:
                raise ValueError('Installed RECORD mismatch: '+name)
            verified += 1
        if not verified: raise ValueError('No hashed files: '+name)
        result[name] = dict(version=dist.version, hashed_files_verified=verified, active_requirements=sorted(requirements))
    return dict(sorted(result.items()))


def dependencies():
    current = catalog_review()
    if current['unresolved_nodes'] != ['track']:
        raise ValueError('Expected only the staged adapter to remain unapproved')
    bindings = {b['node_id']: b for b in current['bindings']}
    graph = build_decomposed_trackml_grid_graph()
    nodes = {n.node_id: n for n in graph.nodes}
    result = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for edge in graph.edges:
            if edge.edge_kind != EdgeKind.CALLABLE_INJECTION:
                continue
            if edge.target_id != 'track':
                raise ValueError('Unexpected callable target')
            binding, node = bindings[edge.source_id], nodes[edge.source_id]
            if not binding['ready']:
                raise ValueError('Numeric dependency must be approved and served')
            contracts = {}
            for prefix in ('atom', 'artifact'):
                version = binding['catalogs'][prefix]['record']['version_id']
                ports = db.execute(f'SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (version,)).fetchall()
                expected = [(direction, p.name, i, p.type_desc) for direction, items in [('input', node.inputs), ('output', node.outputs)] for i,p in enumerate(items)]
                if (prefix == 'artifact' or ports) and [(p['direction'],p['name'],p['ordinal'],p['type_desc']) for p in ports] != expected:
                    raise ValueError('Graph and catalog dependency interface differ: '+node.node_id)
                contracts[prefix] = ports
            if contracts['atom'] and contracts['atom'] != contracts['artifact']:
                raise ValueError('Catalog dependency contracts differ')
            version = binding['catalogs']['artifact']['record']['version_id']
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')", (version,)).fetchone():
                raise ValueError('Failed dependency audit')
            result.append(dict(port_name=edge.input_name, dependency_artifact_fqdn=binding['catalog_fqdn'],
                dependency_content_hash=binding['content_hash'], dependency_role='logic_atom', optional=False,
                runtime_fqdn=binding['runtime_fqdn'], version_id=binding['catalogs']['artifact']['record']['version_id'],
                interfaces=contracts['artifact'], legacy_versioned_interface_present=bool(contracts['atom'])))
    if len(result) != 21:
        raise ValueError('All twenty-one numeric dependencies required')
    return sorted(result, key=lambda r:r['port_name'])


def review():
    intake = reviewed_plan(); staged(intake)
    expected = dependencies()
    paths = list((ROOT/'sciona').glob('trackml_*.py')) + list((ROOT/'sciona').glob('decomposed_trackml_*.py'))
    paths += [ROOT/p for p in ('scripts/review_trackml_grid_adapter_semantics.py',
        'scripts/stage_trackml_grid_adapter.py','scripts/audit_trackml_grid_catalog_readiness.py',
        'scripts/review_trackml_grid_provider_semantics.py','sciona/ghost/registry.py')]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    installed = packages()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        execution = execute_comparison()
    from sciona.atoms.physics.decomposed_trackml_grid_execution import track
    from sciona.ghost.registry import get_witness
    try:
        get_witness(track.__module__+'.'+track.__name__)()
    except ValueError:
        pass
    else:
        raise ValueError('Abstract execution must reject')
    if before != {str(p.relative_to(ROOT)): sha(p) for p in paths} or installed != packages():
        raise ValueError('Qualified recipe software changed')
    if expected != dependencies() or intake != reviewed_plan():
        raise ValueError('Dependency or intake drift')
    staged(intake)
    retained = json.loads((ROOT/'docs/reviews/competition_trackml_lifecycle.json').read_text())
    return dict(format='trackml-grid-adapter-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True,
        adapter=intake['providers'][0], registered_provider_dependencies=expected,
        execution=execution, software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_grid_adapter_intake_plan.json'),
        abstract_witness_rejected=True, resource_policy=dict(serial=True,native_threads=1),
        limitations=retained['limitations']+[
            'Adapter qualifies only the retained configured tracking recipe with the twenty-one exact callables listed here.',
            'Detector topology, candidate lifecycle, retry, ranking, identifier and map dispatch policy remain domain-specific.',
            'Generic providers remain separately selectable; arbitrary replacement callables are not qualified.',
            'Other retained helix helpers and cell arithmetic remain in the recipe and require a separate reuse/scope review before original-CDG publication.',
            'Prepared state and result are concrete runtime objects, not portable serialized artifacts.',
            'Automated Tier3 evidence only; no human Tier1 review or usage-qualified Tier2 claim.',
            'Installed dependency file integrity does not establish publisher authentication or clean deployment on other platforms.',
            'Canonical artifact interfaces define graph dependencies; some legacy atom records intentionally have no versioned interfaces and are preserved.',
            'Original-CDG scope and publication gates remain separate.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_grid_adapter_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True, dependencies=len(result['registered_provider_dependencies']),
        checks=result['execution']['checks'], approved=False)))
