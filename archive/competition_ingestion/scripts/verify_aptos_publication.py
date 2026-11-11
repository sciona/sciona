"""Read back served APTOS versions and execute the catalog graph against the validated derived runtime."""
import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import os
import tempfile
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts import validate_aptos_graph_pipeline as comparison
from scripts import aptos_graph_execution as graph_execution


def verify(root):
    graph_execution.runner._ensure_atoms_imported()
    expected = json.loads((root / 'docs/reviews/competition_aptos_graph_pipeline_validation.json').read_text())
    semantic = json.loads((root / 'docs/reviews/competition_aptos_semantic_review.json').read_text())

    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        fqdn=db.execute('SELECT a.fqdn FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',('94df8d52-0272-5d0b-b84f-1f150d8e11f0',)).fetchone()['fqdn']+'.execution'
        rows = db.execute("SELECT a.artifact_id,v.version_id,v.content_hash,v.trust_tier FROM catalog_artifacts_served a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest", (fqdn,)).fetchall()
        assert len(rows) == 1 and rows[0]['trust_tier'] == 3
        record = rows[0]
        assert record['content_hash'] == expected['graph_execution']['serialized_graph_sha256']
        approval = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='aptos-execution-community.v1' AND passed AND status='completed' AND source_kind='automated'", (record['version_id'],)).fetchall()
        assert len(approval) == 1 and approval[0]['details']['publication_tier'] == 3
        details = approval[0]['details']
        assert details['execution_graph_sha256'] == record['content_hash']
        assert details['limitations'] == semantic['conditions']
        def check_review(identity, legacy=False):
            tables = [('artifact_audit_rollups', 'artifact_id')]
            references = [('artifact_references', 'artifact_id')]
            if legacy:
                tables.append(('atom_audit_rollups', 'atom_id'))
                references.append(('atom_references', 'atom_id'))
            wanted = dict(structural_status='pass', runtime_status='pass',
                semantic_status='conditional_pass', developer_semantics_status='conditional_pass',
                review_semantic_verdict='conditional_pass', review_developer_semantics_verdict='conditional_pass',
                review_status='approved', trust_readiness='ready', review_limitations=semantic['conditions'])
            for table, key in tables:
                row = db.execute('SELECT ' + ','.join(wanted) + ' FROM ' + table + ' WHERE ' + key + '=%s', (identity,)).fetchone()
                assert row == wanted
            for table, key in references:
                rows = db.execute('SELECT r.ref_id,r.url,p.verified FROM ' + table
                    + ' p JOIN references_registry r ON r.ref_id=p.ref_id WHERE p.' + key + '=%s', (identity,)).fetchall()
                assert rows == [dict(ref_id='aptos-winner-eight-model',
                    url='https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary',
                    verified=True)]
        check_review(record['artifact_id'])
        for name, digest in details['evidence_sha256'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
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
            approval = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='aptos-execution-community.v1' AND passed AND status='completed' AND source_kind='automated'", (atom['version_id'],)).fetchall()
            assert len(approval) == 1 and approval[0]['details']['publication_tier'] == 3
            assert approval[0]['details']['runtime_fqdn'] == node.matched_primitive
            assert approval[0]['details']['limitations'] == semantic['conditions']
            check_review(atom['atom_id'], legacy=True)
            reviewed = approval[0]['details']['provider_versions'][node.node_id]
            assert reviewed['version_id'] == str(atom['version_id']) and reviewed['content_hash'] == atom['content_hash']
            assert reviewed['source_sha256'] == hashlib.sha256(Path(inspect.getfile(inspect.unwrap(fn))).read_bytes()).hexdigest()
        outputs = db.execute("SELECT name FROM artifact_io_specs WHERE version_id=%s AND direction='output'", (record['version_id'],)).fetchall()
        assert {r['name'] for r in outputs} == {'result'}
        dependency = db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s', (record['version_id'],)).fetchall()
        assert dependency == [dict(dependency_artifact_fqdn=fqdn.removesuffix('.execution'), dependency_content_hash='02129e9c5bef75cf05490e52f2291f5d336d70a92e114d022a1b635ad35b3f58', dependency_role='cdg', optional=False)]
        source = db.execute('SELECT status,is_publishable FROM artifacts WHERE fqdn=%s', (fqdn.removesuffix('.execution'),)).fetchone()
        assert source == dict(status='draft', is_publishable=False)
        served = db.execute("SELECT count(*) AS n FROM catalog_artifacts_served WHERE artifact_kind='cdg'").fetchone()['n']
    # The numerical validator retains its full budgets and checkpoint checks.
    # Only graph selection and the report sink change; no computation is stubbed.
    evidence_path = root / 'docs/reviews/competition_aptos_graph_pipeline_validation.json'
    before = evidence_path.read_bytes()
    emitted = {}
    original_write = Path.write_text
    def save_report(path, content, *args, **kwargs):
        if path.resolve() == evidence_path.resolve():
            emitted['report'] = json.loads(content)
            return len(content)
        return original_write(path, content, *args, **kwargs)
    def retrieved_graph():
        return graph
    with patch.object(graph_execution, 'build_graph', retrieved_graph), \
         patch.object(Path, 'write_text', save_report):
        comparison.main()
    report = emitted['report']
    assert report['passed'] and report['graph_execution']['actual_runner_nodes'] == 2
    assert report['graph_execution']['serialized_graph_sha256'] == record['content_hash']
    assert report['training_stage_fits'] == 16 and report['second_stage_epochs_per_model'] == 10 and report['optimizer_updates'] == 280
    assert report['checkpoint_replays_exact'] and report['final_ensemble_and_threshold_oracle_passed']
    assert evidence_path.read_bytes() == before
    from scripts.promote_aptos_execution import review
    review(root)
    return dict(read_only=True, synthetic_only=True, trust_tier=3,
        provider_versions_verified=2, served_cdg_count=served,
        catalog_graph_runtime_execution='passed', full_training_and_inference_budgets=True,
        training_stage_fits=16, model_count=8, optimizer_updates=280, graph_sha256=record['content_hash'],
        version_id=str(record['version_id']), original_intake_status='draft',
        served_runtime_validation=report,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = verify(root)
    (root / 'docs/reviews/competition_aptos_publication_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
