"""Qualify corrected first-wave scope for reuse under its original identity."""
import json
import hashlib
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.verify_first_wave_revision_parent import verify
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='693697cf-a97c-53bd-8d6b-19dc3c7697c1'
ARTIFACT='93997609-5796-5a3e-8824-85448fad3dd2'


def require(condition,message):
    if not condition:raise ValueError(message)


def check_parent(db,qualification):
    from scripts.check_first_wave_parent_database import check_parent_database
    graph,parent=check_parent_database(db,ROOT)
    require(parent['version_id']==qualification['version_id'] and parent['graph_sha256']==qualification['graph_sha256'],
            'Qualified parent identity changed')
    return graph,parent


def audit(source):
    qualification=verify(ROOT,source)
    qualification.pop('served_cdg_count') # Unrelated catalog growth is not source qualification.
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph,parent=check_parent(db,qualification)
        row=db.execute('SELECT content_hash FROM artifact_versions WHERE version_id=%s AND artifact_id=%s',(ORIGINAL,ARTIFACT)).fetchone()
    require(len(graph.nodes)==1 and not graph.edges,'Complete corrected dynamics graph required')
    require(graph.metadata['source_version_id']==ORIGINAL and graph.metadata['source_content_hash']==row['content_hash'],'Original source identity differs')
    return dict(read_only=True,approved=False,catalog_mutations=0,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_checker_sha256=hashlib.sha256((ROOT/'scripts/check_first_wave_parent_database.py').read_bytes()).hexdigest(),graphs=[dict(family='first_wave_dynamics',
        legacy_artifact_id=ARTIFACT,legacy_version_id=ORIGINAL,legacy_content_hash=row['content_hash'],
        projected_source_version_id=ORIGINAL,approved_execution_version_id=parent['version_id'],
        approved_execution_graph_sha256=parent['graph_sha256'],qualification=qualification,
        corrections=graph.metadata['corrections'],assumptions=graph.metadata['assumptions'],exclusions=graph.metadata['exclusions'],
        scope=graph.metadata['scope'],new_atoms_required=0)],
        limitations=['Corrected differential dynamics with explicitly added kinematic premise and positive constant mass.',
            'Historical derivative parse remains invalid and preserved. This is not literal proof parity or approval.',
            'Reusable symbolic differentiation and point evaluation in a stated inertial Cartesian regime; no time integration or unqualified cross-domain transfer.'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-directory',type=Path,required=True)
    report=audit(p.parse_args().source_directory)
    (ROOT/'docs/reviews/first_wave_identity_execution_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=False,graphs=1,new_atoms=0)))
