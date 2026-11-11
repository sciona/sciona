"""Assess complete corrected Otto scope under the preserved original identity.

No staging, approval or fresh native-fit claim is made by this read-only audit.
"""
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.review_otto_execution import audit,STAGE_MAPPING,SOURCE_VERSION,SOURCE_HASH
from scripts.check_otto_identity_parent import check_parent
from scripts.stage_remaining_legacy_physics_revisions import history
from sciona.competition_import import _canonical

ROOT=Path(__file__).resolve().parents[1]


def review():
    qualification=audit()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        rows=db.execute("SELECT a.artifact_id::text,a.status,a.is_publishable,v.content_hash,v.is_latest,e.details FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id) WHERE v.version_id=%s AND e.runner_version='competition-intake.v1'",(SOURCE_VERSION,)).fetchall()
        if len(rows)!=1:raise ValueError('One preserved Otto intake required')
        row=rows[0];snapshot=row['details']['snapshot']
        if row['content_hash']!=SOURCE_HASH or hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=SOURCE_HASH:
            raise ValueError('Original intake snapshot changed')
        stages=[s['stage_id'] for s in snapshot['template']['stages']]
        if len(stages)!=6 or set(stages)!=set(STAGE_MAPPING):raise ValueError('Complete six-stage correction mapping required')
        graph,parent=check_parent(db,ROOT)
        if len(graph.nodes)!=2 or parent['graph_sha256']!=qualification['serialized_graph_sha256']:
            raise ValueError('Complete qualified parent graph required')
        preserved=history(db,SOURCE_VERSION)
        current=db.execute('SELECT version_id::text,derives_from::text,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',(row['artifact_id'],)).fetchall()
        if row['status']=='draft':
            if row['is_publishable'] or not row['is_latest'] or len(current)!=1 or current[0]['version_id']!=SOURCE_VERSION:
                raise ValueError('Draft original selection differs')
        elif row['status']=='approved':
            if not row['is_publishable'] or row['is_latest'] or len(current)!=1 or current[0]['derives_from']!=SOURCE_VERSION or current[0]['trust_tier']!=3:
                raise ValueError('Corrected original selection differs')
        else:raise ValueError('Unsupported original state')
    return dict(read_only=True,catalog_mutations=0,eligible_for_approval=False,artifact_id=row['artifact_id'],
        original_version_id=SOURCE_VERSION,original_content_hash=SOURCE_HASH,original_history_sha256=preserved,
        qualified_parent=parent,retained_semantic_qualification=qualification,complete_stage_corrections=STAGE_MAPPING,
        reuse_scope='Candidate nine-class tabular stacking method; the retained two-node prepare/execute wrapper does not establish compositional reuse.',
        compositional_reuse_qualified=False,
        required_before_promotion=['Decompose the opaque wrapper into independently callable computational operations with explicit contracts and domain adapters, exposing first-level producers, supplemental features, meta selection/refits and final combination.',
            'Qualify transferable operations with independent synthetic examples from materially different domains; retain nine-class, fold, transductive-population and source-policy constraints in explicit adapters.',
            'Stage the decomposed corrected graph under original identity with mandatory source/parent provenance and exact qualified provider bindings; copying the existing two-node graph is insufficient.',
            'Execute the corrected stored graph with all 33 first-level entries and complete 8800 meta tuning fits plus 1100 refits on synthetic inputs under a verified shared CPU budget.',
            'Verify candidate/parent corruption rejection, activation rollback, idempotence and fresh served selection.'],
        limitations=['Retained evidence is freshly integrity-checked; this audit does not rerun native training.',
            'Historical feature-width and implementation ambiguities remain explicit; no historical winner accuracy or arbitrary class-count claim.',
            'The existing semantic policy says original intake stays draft; an original-identity publication must explicitly qualify the corrected revision while preserving that historical policy and snapshot.'],
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['scripts/audit_otto_original_identity_scope.py','scripts/check_otto_identity_parent.py','scripts/review_otto_execution.py']})


if __name__=='__main__':
    try:
        report=review()
    except (OSError, AssertionError) as exc:
        # Never leave a prior successful report looking like this run passed.
        # No database mutations occur anywhere in this audit.
        report=dict(read_only=True,catalog_mutations=0,eligible_for_approval=False,
                    compositional_reuse_qualified=False,status='qualification_failed',
                    failure_type=type(exc).__name__,failure_detail=str(exc),
                    required_before_promotion=[
                        'Restore or explicitly requalify the native runtime dependency closure; retained qualification did not pass the fresh audit.',
                        'Decompose the prepare/execute wrapper into reusable operations and explicit source-policy adapters.',
                        'Qualify the complete corrected graph and its providers before staging or promotion.'],
                    auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (ROOT/'docs/reviews/competition_otto_original_identity_scope.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
        raise SystemExit(1)
    (ROOT/'docs/reviews/competition_otto_original_identity_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(read_only=True,stage_corrections=len(report['complete_stage_corrections']),qualified_parent=report['qualified_parent'],eligible_for_approval=False)))
