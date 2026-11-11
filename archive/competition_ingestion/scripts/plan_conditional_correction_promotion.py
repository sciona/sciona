"""Read-only catalog preflight; never approves or imports artifacts."""
import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import sciona.atoms.ml.calibration.conditional_residuals as provider
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.conditional_correction_graph import build_conditional_correction_graph
from scripts.review_conditional_correction import ROOT, SOURCE_HASH, SOURCE_VERSION, audit, require, sha

FQDN = 'cdg.ml.calibration.conditional_residual_correction'


def plan():
    semantic = audit()
    graph = build_conditional_correction_graph()
    directory = ROOT.parent / 'sciona-atoms-ml'
    inventory = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                         artifact_root=directory / 'src/sciona/atoms')
    bindings = []
    for node in graph.nodes:
        matches = [s for s in inventory if s.import_module + '.' + s.source_symbol == node.matched_primitive]
        require(len(matches) == 1, 'Unique provider inventory identity required')
        spec = matches[0]
        function = getattr(provider, spec.source_symbol)
        require(Path(inspect.getfile(inspect.unwrap(function))).resolve() == spec.file_path.resolve(), 'Provider import shadowed')
        require(sha(spec.file_path) == semantic['provider_sha256'], 'Provider identity drift')
        bindings.append(dict(node_id=node.node_id, fqdn=spec.fqdn,
            artifact_id=str(uuid5(NAMESPACE_URL, 'sciona-provider-draft:' + spec.fqdn)),
            version_id=str(spec.version_id), content_hash=spec.content_hash,
            runtime_fqdn=node.matched_primitive,
            inputs=[p.model_dump() for p in node.inputs], outputs=[p.model_dump() for p in node.outputs]))
    identity = uuid5(NAMESPACE_URL, 'sciona-reusable-cdg:' + FQDN)
    version = uuid5(identity, semantic['serialized_graph_sha256'])
    graph_target = dict(fqdn=FQDN, artifact_id=str(identity), version_id=str(version),
                        content_hash=semantic['serialized_graph_sha256'])
    # Read-only transaction makes accidental writes impossible in this preflight.
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
            row_factory=dict_row, options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        source = db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash FROM artifacts a '
            'JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        require(source is not None and source['content_hash'] == SOURCE_HASH, 'Original intake identity differs')
        require(source['status'] == 'draft' and not source['is_publishable'], 'Original intake state differs')
        for target in bindings + [graph_target]:
            found = db.execute('SELECT artifact_id,status,is_publishable FROM artifacts WHERE fqdn=%s', (target['fqdn'],)).fetchall()
            require(len(found) <= 1, 'Duplicate catalog identity')
            if found:
                require(str(found[0]['artifact_id']) == target['artifact_id'], 'Conflicting canonical identity')
            target['existing_catalog_state'] = [dict(artifact_id=str(r['artifact_id']), status=r['status'],
                                                      is_publishable=r['is_publishable']) for r in found]
            versions = db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s',
                                  (target['version_id'],)).fetchall()
            require(all(str(r['artifact_id']) == target['artifact_id'] and r['content_hash'] == target['content_hash']
                        for r in versions), 'Conflicting canonical version')
        for target in bindings:
            legacy = db.execute('SELECT atom_id,status,is_publishable FROM atoms WHERE fqdn=%s', (target['fqdn'],)).fetchall()
            require(len(legacy) <= 1 and all(str(r['atom_id']) == target['artifact_id'] for r in legacy),
                    'Conflicting legacy atom identity')
            target['existing_legacy_state'] = [dict(atom_id=str(r['atom_id']), status=r['status'],
                                                  is_publishable=r['is_publishable']) for r in legacy]
            versions = db.execute('SELECT atom_id,content_hash FROM atom_versions WHERE version_id=%s',
                                  (target['version_id'],)).fetchall()
            require(all(str(r['atom_id']) == target['artifact_id'] and r['content_hash'] == target['content_hash']
                        for r in versions), 'Conflicting legacy version')
        provider_owners = db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a "
            "JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        require(len(provider_owners) == 1, 'Ambiguous provider ownership')
    connected = {(e.target_id, e.input_name) for e in graph.edges}
    consumed = {(e.source_id, e.output_name) for e in graph.edges}
    inputs, outputs = {}, {}
    for node in graph.nodes:
        for ports, excluded, boundary in [(node.inputs, connected, inputs), (node.outputs, consumed, outputs)]:
            for port in ports:
                if (node.node_id, port.name) in excluded:
                    continue
                require(port.name not in boundary or boundary[port.name] == port.model_dump(), 'Ambiguous boundary')
                boundary[port.name] = port.model_dump()
    require(len(inputs) == 6 and len(outputs) == 1, 'Unexpected boundary arity')
    return dict(format='conditional-correction-promotion-plan.v1', catalog_mutations=0, approved=False,
        fqdn=FQDN, artifact_id=str(identity), version_id=str(version), trust_tier=3,
        graph_sha256=semantic['serialized_graph_sha256'], bindings=bindings,
        existing_graph_state=graph_target['existing_catalog_state'],
        boundary_inputs=list(inputs.values()), boundary_outputs=list(outputs.values()),
        mandatory_provenance=dict(fqdn=source['fqdn'], version_id=SOURCE_VERSION, content_hash=SOURCE_HASH,
                                 scope='Original conceptual intake provenance; partial correction subgraph only'),
        publication_blockers=semantic['publication_blockers'], planner_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    result = plan()
    (ROOT / 'docs/reviews/conditional_correction_promotion_plan.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(fqdn=result['fqdn'], providers=len(result['bindings']), inputs=len(result['boundary_inputs']),
                          outputs=len(result['boundary_outputs']), catalog_mutations=0)))
