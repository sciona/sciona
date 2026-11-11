"""Read-only verification of the corrected original CDG and all pinned bindings."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.nasa_first_complete_graph import build_nasa_first_complete_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_first_corrected_intake import ROOT,plan,sha,require,check_source_anchor,check_atoms,origin_rows
from scripts.import_nasa_first_corrected_intake import RUNNER


def check_candidate(db,proposed):
    check_source_anchor(db);check_atoms(db,proposed['atoms'])
    require(origin_rows(db)==proposed['original_rows_sha256'],'Original historical records changed')
    version=proposed['version_id']
    state=db.execute('SELECT a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(version,)).fetchone()
    require(state==dict(status='draft',is_publishable=False,content_hash=proposed['graph_sha256'],is_latest=False,trust_tier=3),'Candidate selection state differs')
    require(db.execute('SELECT is_latest FROM artifact_versions WHERE version_id=%s',(proposed['original_version_id'],)).fetchone()['is_latest'],'Original latest version changed')
    columns=['direction','name','ordinal','type_desc','constraints','required','default_value_repr','dim_signature']
    actual=db.execute(f'SELECT {",".join(columns)} FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()
    expected=[]
    for direction,ports in [('input',proposed['inputs']),('output',proposed['outputs'])]:
        for ordinal,port in enumerate(ports):
            expected.append({key:direction if key=='direction' else ordinal if key=='ordinal' else port[key] for key in columns})
    require(actual==expected,'Candidate root contracts differ')
    bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,binding_confidence,binding_source,status,alternatives,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(version,)).fetchall()
    expected=[dict(node_id=b['node_id'],bound_artifact_fqdn=b['fqdn'],bound_version_content_hash=b['content_hash'],binding_confidence=1.,binding_source=RUNNER,
        status='active',alternatives=[],evidence_summary=dict(runtime_fqdn=b['runtime_fqdn'],provider_version_id=b['version_id'],output_aliases_by_ordinal=[p['name'] for p in b['outputs']]))
        for b in sorted(proposed['bindings'],key=lambda b:b['node_id'])]
    require(sorted(bindings,key=lambda r:r['node_id'])==expected,'Candidate node bindings differ')
    dependencies=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,port_name,dependency_role,optional,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(version,)).fetchall()
    expected=[dict(dependency_artifact_fqdn=proposed['fqdn'],dependency_content_hash=proposed['original_content_hash'],port_name='',dependency_role='cdg',optional=False,
        binding_metadata=dict(source_version_id=proposed['original_version_id'],scope='Immutable source/version provenance; runtime computation is expressed by direct atom bindings.'))]
    expected.extend(dict(dependency_artifact_fqdn=a['fqdn'],dependency_content_hash=a['content_hash'],port_name='',dependency_role='logic_atom',optional=False,
        binding_metadata=dict(provider_version_id=a['version_id'],runtime_fqdn=a['runtime_fqdn'],scope='Approved runtime dependency, including nested branch and controller-injected operations.')) for a in proposed['atoms'])
    require(sorted(dependencies,key=lambda r:r['dependency_artifact_fqdn'])==sorted(expected,key=lambda r:r['dependency_artifact_fqdn']),'Candidate dependency closure differs')
    audit=db.execute('SELECT passed,status,source_kind,details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',(version,RUNNER)).fetchall()
    require(audit==[dict(passed=True,status='completed',source_kind='automated',details=dict(
        scope='Corrected execution version staged without approving or rewriting the original snapshot.',original_rows_sha256=proposed['original_rows_sha256'],
        evidence_sha256=proposed['evidence_sha256'],planner_sha256=proposed['planner_sha256'],nested_bindings=proposed['nested_bindings'],
        controller_dependency=proposed['controller_dependency'],output_bindings=proposed['output_bindings']))],'Candidate staging evidence differs')
    require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(version,)).fetchone(),'Candidate failed evidence remains')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
    graph=_artifact_document_to_cdg(document,version_id=version,content_hash=proposed['graph_sha256'],require_execution_envelope=True)
    require(graph==build_nasa_first_complete_graph(),'Stored candidate topology differs')
    require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchone(),'Draft unexpectedly served')
    return graph


def verify():
    proposed=plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=check_candidate(db,proposed)
    return dict(passed=True,approved=False,catalog_mutations=0,read_only=True,version_id=proposed['version_id'],graph_sha256=proposed['graph_sha256'],
        nodes=len(graph.nodes),edges=len(graph.edges),node_bindings=len(proposed['bindings']),approved_dependencies=len(proposed['atoms']),
        original_history_preserved=True,original_failed_preflight_preserved=True,version_selected_production_converter=True,
        verifier_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/reviews/competition_nasa_first_corrected_intake_plan.json'))


if __name__=='__main__':
    report=verify()
    (ROOT/'docs/reviews/competition_nasa_first_corrected_draft_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','nodes','node_bindings','approved_dependencies','original_history_preserved']}))
