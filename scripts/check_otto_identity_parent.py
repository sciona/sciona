"""Exact served Otto parent checks inside a caller-owned read-only transaction."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph


def check_parent(db,root):
    expected=json.loads((root/'docs/reviews/competition_otto_graph_execution.json').read_text())
    fqdn=db.execute('SELECT a.fqdn FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',('dc4969f7-e5a4-5644-a9d0-fa10493458ed',)).fetchone()['fqdn']+'.execution'
    rows = db.execute("SELECT a.artifact_id,v.version_id,v.content_hash,v.trust_tier FROM catalog_artifacts_served a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest", (fqdn,)).fetchall()
    assert len(rows) == 1 and rows[0]['trust_tier'] == 3
    record = rows[0]
    assert record['content_hash'] == expected['serialized_graph_sha256']
    approval = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='otto-execution-community.v1' AND passed AND status='completed' AND source_kind='automated'", (record['version_id'],)).fetchall()
    assert len(approval) == 1 and approval[0]['details']['publication_tier'] == 3
    details = approval[0]['details']
    assert details['execution_graph_sha256'] == record['content_hash']
    for name, digest in details['evidence_sha256'].items():
        assert hashlib.sha256((root / 'docs/reviews' / name).read_bytes()).hexdigest() == digest
    for name, digest in details['implementation_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    doc = db.execute('SELECT get_artifact_document(%s) AS d', (fqdn,)).fetchone()['d']
    graph = _artifact_document_to_cdg(doc, version_id=str(record['version_id']), content_hash=record['content_hash'], require_execution_envelope=True)
    assert encode_execution_graph(graph)[0] == record['content_hash']
    bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (record['version_id'],)).fetchall()
    assert len(bindings) == len(graph.nodes) == 2
    for binding in bindings:
        assert binding['status'] == 'active'
        node = next(n for n in graph.nodes if n.node_id == binding['node_id'])
        atom = db.execute("SELECT a.atom_id,p.import_module,p.source_symbol,v.version_id,v.content_hash,v.trust_tier FROM catalog_atoms_served a JOIN atoms p USING(atom_id) JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest", (binding['bound_artifact_fqdn'],)).fetchall()
        assert len(atom) == 1
        atom = atom[0]
        assert atom['content_hash'] == binding['bound_version_content_hash'] and atom['trust_tier'] == 3
        assert node.matched_primitive == atom['import_module'] + '.' + atom['source_symbol']
        assert db.execute('SELECT 1 FROM catalog_artifacts_served a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=%s AND v.version_id=%s AND v.is_latest AND v.content_hash=%s AND v.trust_tier=3', (atom['atom_id'], atom['version_id'], atom['content_hash'])).fetchone()
        fn = getattr(importlib.import_module(atom['import_module']), atom['source_symbol'])
        for table in ['artifact_io_specs', 'atom_io_specs']:
            ports = db.execute('SELECT name,required,default_value_repr FROM ' + table + " WHERE version_id=%s AND direction='input' ORDER BY ordinal", (atom['version_id'],)).fetchall()
            wanted = [dict(name=name, required=p.default is p.empty, default_value_repr='' if p.default is p.empty else repr(p.default)) for name, p in inspect.signature(fn).parameters.items()]
            assert ports == wanted
        approval = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='otto-execution-community.v1' AND passed AND status='completed' AND source_kind='automated'", (atom['version_id'],)).fetchall()
        assert len(approval) == 1 and approval[0]['details']['publication_tier'] == 3
        assert approval[0]['details']['runtime_fqdn'] == node.matched_primitive
        reviewed = approval[0]['details']['provider_versions'][node.node_id]
        assert reviewed['version_id'] == str(atom['version_id']) and reviewed['content_hash'] == atom['content_hash']
        assert reviewed['source_sha256'] == hashlib.sha256(Path(inspect.getfile(inspect.unwrap(fn))).read_bytes()).hexdigest()
    outputs = db.execute("SELECT name FROM artifact_io_specs WHERE version_id=%s AND direction='output'", (record['version_id'],)).fetchall()
    assert {r['name'] for r in outputs} == {'result'}
    dependency = db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s', (record['version_id'],)).fetchall()
    assert dependency == [dict(dependency_artifact_fqdn=fqdn.removesuffix('.execution'), dependency_content_hash='13be8fd78b963daf9fd05fb5758e1d36fa98d4ed8cfdde85eb829c94f73af30a', dependency_role='cdg', optional=False)]
    source = db.execute('SELECT status,is_publishable FROM artifacts WHERE fqdn=%s', (fqdn.removesuffix('.execution'),)).fetchone()
    assert source in [dict(status='draft',is_publishable=False),dict(status='approved',is_publishable=True)]
    served = db.execute("SELECT count(*) AS n FROM catalog_artifacts_served WHERE artifact_kind='cdg'").fetchone()['n']
    return graph,dict(version_id=str(record['version_id']),graph_sha256=record['content_hash'])
