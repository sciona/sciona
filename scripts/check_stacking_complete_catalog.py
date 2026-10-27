"""Exact stored-contract checks for the staged complete Otto revision.

Caller owns the transaction, allowing faults to be checked before rollback.
This verifies catalog integrity, not numerical execution or publication evidence.
"""
import json
from uuid import UUID, NAMESPACE_URL, uuid5

from sciona.architect.models import IOSpec
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.stacking_complete_graph import build_complete_graph
from scripts.import_residual_execution_drafts import comparable
from scripts.promote_otto_execution import boundary_ports
from scripts.review_stacking_complete_scope import ROOT, review, require, sha
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts.stage_stacking_complete import RUNNER

ORIGINAL_FQDN = 'cdg.competition.solution.kaggle.otto_group_product_1st'


def qualification():
    scope = review()
    require(scope==json.loads((ROOT/'docs/reviews/competition_otto_decomposed_complete_scope.json').read_text()),
            'Retained scope differs')
    directory = ROOT.parent/'sciona-atoms-ml'
    parsed = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',directory),
                                     artifact_root=directory/'src/sciona/atoms')
    specs = {s.fqdn:s for s in parsed if s.fqdn in scope['providers']}
    require(len(specs)==27, 'Provider inventory differs')
    return scope, specs


def check_ports(db, table, owner_key, owner, version, contract, message):
    columns = ['direction','ordinal','name','type_desc','constraints','required','default_value_repr']
    if table=='artifact_io_specs':
        columns.append('dim_signature')
    actual = db.execute('SELECT '+owner_key+'::text,'+','.join(columns)+' FROM '+table+
                        ' WHERE version_id=%s ORDER BY direction,ordinal', (version,)).fetchall()
    expected = []
    for direction in ('input','output'):
        for ordinal, port in enumerate(contract[direction]):
            row = {k:port[k] for k in columns if k not in ('direction','ordinal')}
            row.update(direction=direction,ordinal=ordinal,**{owner_key:str(owner)})
            expected.append(row)
    require(actual==expected, message)


def check_evidence(db, owner, version, details, message):
    rows = db.execute('SELECT evidence_id::text,artifact_id::text,audit_type,passed,status,source_kind,details '
                      'FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',
                      (version,RUNNER)).fetchall()
    require(rows==[dict(evidence_id=str(uuid5(UUID(str(version)),RUNNER)),artifact_id=str(owner),
                        audit_type='asset_integrity_check',passed=True,status='completed',
                        source_kind='automated',details=details)], message)
    require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',
                           (version,)).fetchone(), 'Unresolved target audit failure')


def check_candidate(db, scope, specs, *, approved=False):
    identity, version, original = scope['artifact_id'], scope['corrected_version_id'], scope['original_version_id']
    state = db.execute('SELECT fqdn,artifact_kind,status,is_publishable FROM artifacts WHERE artifact_id=%s',
                       (identity,)).fetchone()
    require(state==dict(fqdn=ORIGINAL_FQDN,artifact_kind='cdg',status='approved' if approved else 'draft',
                       is_publishable=approved), 'Original identity state differs')
    require(history(db,original)==scope['original_history_sha256'], 'Original history differs')
    row = db.execute('SELECT artifact_id::text,content_hash,semver,is_latest,trust_tier,derives_from::text '
                     'FROM artifact_versions WHERE version_id=%s', (version,)).fetchone()
    require(row==dict(artifact_id=identity,content_hash=scope['graph_sha256'],
                      semver='0.1.0+decomposed.'+scope['graph_sha256'][:12],is_latest=approved,trust_tier=3,
                      derives_from=original), 'Corrected version differs')
    require(db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',
                       (identity,)).fetchall()==[dict(version_id=version if approved else original)],
            'Original latest selection differs')
    graph = build_complete_graph()
    by_runtime = {p['runtime_fqdn']:fqdn for fqdn,p in scope['providers'].items()}
    interfaces = {n.node_id:{d:[IOSpec.model_validate(p) for p in ps]
                            for d,ps in scope['providers'][by_runtime[n.matched_primitive]]['ports'].items()}
                  for n in graph.nodes}
    boundary = {d:[p.model_dump(mode='json') for p in ps] for d,ps in boundary_ports(graph,interfaces).items()}
    check_ports(db,'artifact_io_specs','artifact_id',identity,version,boundary,'Corrected root contracts differ')
    actual = db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,binding_confidence,'
                        'binding_source,status,alternatives,evidence_summary FROM artifact_cdg_bindings '
                        'WHERE version_id=%s', (version,)).fetchall()
    expected = []
    for node in graph.nodes:
        fqdn = by_runtime[node.matched_primitive]; p = scope['providers'][fqdn]
        expected.append(dict(node_id=node.node_id,bound_artifact_fqdn=fqdn,bound_version_content_hash=p['content_hash'],
            binding_confidence=1.,binding_source=RUNNER,status='active',alternatives=[],evidence_summary=dict(
                runtime_fqdn=node.matched_primitive,provider_version_id=p['version_id'],source_sha256=p['source_sha256'],
                output_aliases_by_ordinal=[p.name for p in node.outputs])))
    require(sorted(actual,key=lambda r:r['node_id'])==sorted(expected,key=lambda r:r['node_id']),
            'Corrected bindings differ')
    actual = db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,port_name,'
                        'binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s', (version,)).fetchall()
    require(actual==[dict(dependency_artifact_fqdn=ORIGINAL_FQDN,dependency_content_hash=scope['original_content_hash'],
        dependency_role='cdg',optional=False,port_name='',binding_metadata=dict(source_version_id=original,
            scope='Immutable original intake provenance; numerical execution uses direct atom bindings.'))],
        'Corrected provenance differs')
    check_evidence(db,identity,version,dict(scope=scope,approved=False,stager_sha256=sha(ROOT/'scripts/stage_stacking_complete.py')),
                   'Corrected staging evidence differs')
    document = db.execute('SELECT get_artifact_document(%s) AS d',(ORIGINAL_FQDN,)).fetchone()['d']
    require(_artifact_document_to_cdg(document,version_id=version,content_hash=scope['graph_sha256'],
                                     require_execution_envelope=True)==graph, 'Stored graph differs')
    owners = [identity]
    for fqdn,p in sorted(scope['providers'].items()):
        spec = specs[fqdn]; owner = str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+fqdn)); owners.append(owner)
        require(str(spec.version_id)==p['version_id'] and spec.content_hash==p['content_hash']
                and sha(spec.file_path)==p['source_sha256'], 'Live provider differs')
        for table, key in [('artifacts','artifact_id'),('atoms','atom_id')]:
            columns = ['fqdn','status','is_publishable']
            expected = dict(fqdn=fqdn,
                            status='approved' if approved else ('draft' if table=='artifacts' else 'flagged'),
                            is_publishable=approved)
            if table=='artifacts':
                columns.append('artifact_kind'); expected['artifact_kind']='atom'
            else:
                columns.extend(['import_module','source_symbol'])
                expected.update(import_module=spec.import_module,source_symbol=spec.source_symbol)
            require(db.execute('SELECT '+','.join(columns)+' FROM '+table+' WHERE '+key+'=%s',
                               (owner,)).fetchone()==expected, 'Provider identity differs: '+table)
        for table,key in [('artifact_versions','artifact_id'),('atom_versions','atom_id')]:
            rows = db.execute('SELECT '+key+'::text,content_hash,semver,is_latest,trust_tier,s3_key,fingerprint '
                              'FROM '+table+' WHERE version_id=%s',(p['version_id'],)).fetchall()
            require(comparable(rows)==comparable([dict(**{key:owner},content_hash=p['content_hash'],semver=spec.semver,
                        is_latest=True,trust_tier=3,s3_key='',fingerprint=spec.fingerprint)]),
                    'Provider version differs: '+table)
            require(db.execute('SELECT version_id::text FROM '+table+' WHERE '+key+'=%s AND is_latest',
                               (owner,)).fetchall()==[dict(version_id=p['version_id'])], 'Provider latest selection differs')
        for table,key in [('artifact_io_specs','artifact_id'),('atom_io_specs','atom_id')]:
            check_ports(db,table,key,owner,p['version_id'],p['ports'],'Provider ports differ: '+table)
        check_evidence(db,owner,p['version_id'],dict(scope='Exact provider identity and ports staged; not approval.',
            provider=p,scope_sha256=sha(ROOT/'docs/reviews/competition_otto_decomposed_complete_scope.json')),
            'Provider staging evidence differs')
        require(not db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',
                               (p['version_id'],)).fetchone(), 'Unexpected provider dependency')
    served = db.execute('SELECT DISTINCT artifact_id::text FROM catalog_artifacts_served WHERE artifact_id=ANY(%s::uuid[])',
                        (owners,)).fetchall()
    require({r['artifact_id'] for r in served}==(set(owners) if approved else set()), 'Candidate serving state differs')
    return graph
