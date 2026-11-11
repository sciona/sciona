"""Read-only source, implementation and catalog readiness for the TrackML original."""
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from scripts.import_flavours_provider_drafts import ROOT,connect
from sciona.competition_import import _canonical

IDENTITY='a2067934-0f90-5083-81b3-71025d158b6a'
ORIGINAL='11c41ece-c648-50b4-b211-7e319ba36dea'
SOURCE_HASH='c0b732d94cd161a98dfc8a24a70a1805453a3de2d5b4d71ee7719fda4c639183'
DERIVATIVE='249c8ff0-9e0a-5ccb-823c-2482f8976592'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    reports={};checks={};drift=[]
    for path in sorted((ROOT/'docs/reviews').glob('competition_trackml_*.json')):
        reports[path.name]=sha(path)
        report=json.loads(path.read_text())
        if not isinstance(report,dict):continue
        for section in ('code_sha256','implementation_sha256','hashes'):
            for relative,digest in report.get(section,{}).items():
                if (relative,digest) in checks:continue
                current=sha(ROOT/relative) if (ROOT/relative).is_file() else None
                checks[relative,digest]=current
                if current!=digest:drift.append(dict(file=relative,recorded=digest,current=current))
    inventory={}
    for path in sorted((ROOT/'sciona').glob('trackml_*.py')):
        tree=ast.parse(path.read_text())
        inventory[path.name]=dict(sha256=sha(path),functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))])
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s''',(ORIGINAL,)).fetchone()
        if not original or original['artifact_id']!=IDENTITY or original['content_hash']!=SOURCE_HASH:
            raise ValueError('Original identity changed')
        evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
        if len(evidence)!=1 or hashlib.sha256(_canonical(evidence[0]['details']['snapshot']).encode()).hexdigest()!=SOURCE_HASH:
            raise ValueError('Immutable intake snapshot required')
        stages=[dict(stage_id=s['stage_id'],description=s['description']) for s in evidence[0]['details']['snapshot']['template']['stages']]
        derivative=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s''',(DERIVATIVE,)).fetchone()
        if not derivative or derivative['status']!='approved' or not derivative['is_publishable'] or not derivative['is_latest'] or derivative['trust_tier']!=3:
            raise ValueError('Approved reference derivative required')
        nodes=db.execute('SELECT node_id FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(DERIVATIVE,)).fetchall()
        bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(DERIVATIVE,)).fetchall()
        served=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s',(derivative['artifact_id'],)).fetchone()['n']
        if served!=1 or len(nodes)!=3 or len(bindings)!=3:raise ValueError('Complete served three-stage reference required')
        remaining=db.execute("""SELECT count(DISTINCT a.artifact_id) AS n FROM artifacts a
            WHERE a.status<>'approved' AND EXISTS(SELECT 1 FROM artifact_audit_evidence e WHERE e.artifact_id=a.artifact_id AND e.runner_version='competition-intake.v1')
            AND EXISTS(SELECT 1 FROM artifacts d WHERE d.fqdn LIKE a.fqdn||'.%' AND d.status='approved')""").fetchone()['n']
    return dict(format='decomposed-trackml-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        original=original,source_version_id=ORIGINAL,source_stages=stages,derivative=derivative,
        derivative_version_id=DERIVATIVE,derivative_nodes=nodes,derivative_bindings=bindings,
        draft_originals_with_approved_derivatives=remaining,retained_evidence_sha256=reports,
        distinct_file_hash_checks=len(checks),implementation_drift=drift,implementation_inventory=inventory,
        current_runtime_versions={n:metadata.version(n) for n in ('numpy','scipy','pandas','scikit-learn')},
        source_corrections=[
            'Existing three-node derivative implements iterative tracking with caller-supplied detector geometry and optional maps; its declared exclusions include learned calibration, sparse-layer tracking, truth scoring and file outputs.',
            'Original eleven-stage scope must be compared with all retained source implementations; an approved tracking derivative alone cannot establish original completion.',
            'Empty-candidate and no-seed compatibility changes are explicit; no historical competition accuracy claim.'],
        reuse_work=[
            'Separate generic circle/helix geometry, interpolation and neighbor operations from detector topology, track-candidate policy and calibration lifecycle.',
            'Inspect registered providers before adding implementations and preserve pinned BSD source attribution.',
            'Expose complete original computation and source-required learning stages, not only decode/track/encode wrappers.',
            'Revalidate current source comparisons and exact catalog bindings before promotion.'],
        limits=['Readiness only; historical hashes do not establish current numerical execution.',
                'Public algorithm-template descriptions and software identifiers only; runtime records and identifying dataset metadata excluded.'],
        auditor_sha256=sha(__file__))


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/decomposed_trackml_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(original=result['original']['status'],reference_nodes=len(result['derivative_nodes']),
        checked=result['distinct_file_hash_checks'],drift=len(result['implementation_drift']),remaining=result['draft_originals_with_approved_derivatives'])))
