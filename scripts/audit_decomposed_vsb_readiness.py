"""Read-only source, implementation and catalog readiness for the VSB original."""
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from scripts.import_flavours_provider_drafts import ROOT,connect
from sciona.competition_import import _canonical

IDENTITY='905df5de-530b-5394-b8e0-a355c8f1c3ce'
ORIGINAL='d79e40e4-45b0-5e74-a83c-c68c91f12afb'
SOURCE_HASH='565e511ee8d20ce0b32adf5c8cc726150f235b1f97d5502d62c96cfe339f5ac1'
DERIVATIVE='feab5779-925c-5fe0-850b-c865942833ba'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    reports={};checks={};drift=[]
    for path in sorted((ROOT/'docs/reviews').glob('competition_vsb_*.json')):
        reports[path.name]=sha(path)
        report=json.loads(path.read_text())
        for section in ('code_sha256','implementation_sha256'):
            for relative,digest in report.get(section,{}).items():
                if (relative,digest) in checks:continue
                current=sha(ROOT/relative) if (ROOT/relative).is_file() else None
                checks[relative,digest]=current
                if current!=digest:drift.append(dict(file=relative,recorded=digest,current=current))
    inventory={}
    for path in sorted((ROOT/'sciona').glob('vsb_*.py')):
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
        if served!=1 or len(nodes)!=2 or len(bindings)!=2:raise ValueError('Complete served two-stage reference required')
        remaining=db.execute("""SELECT count(DISTINCT a.artifact_id) AS n FROM artifacts a
            WHERE a.status<>'approved' AND EXISTS(SELECT 1 FROM artifact_audit_evidence e WHERE e.artifact_id=a.artifact_id AND e.runner_version='competition-intake.v1')
            AND EXISTS(SELECT 1 FROM artifacts d WHERE d.fqdn LIKE a.fqdn||'.%' AND d.status='approved')""").fetchone()['n']
    return dict(format='decomposed-vsb-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        original=original,source_version_id=ORIGINAL,source_stages=stages,derivative=derivative,
        derivative_version_id=DERIVATIVE,derivative_nodes=nodes,derivative_bindings=bindings,
        draft_originals_with_approved_derivatives=remaining,retained_evidence_sha256=reports,
        distinct_file_hash_checks=len(checks),implementation_drift=drift,implementation_inventory=inventory,
        current_runtime_versions={n:metadata.version(n) for n in ('numpy','scipy','pandas','lightgbm','scikit-learn')},
        source_corrections=[
            'Independent reconstruction replaces unsupported wavelet, image transform, deep network and weighted tree/deep blend with recursive baseline, custom peaks, phase aggregates and equal repeated-tree probabilities.',
            'Preserve all 25 repetitions and five folds with 10000-round ceiling and 100-round stopping patience; both heldouts influence stopping.',
            'Threshold is fitted to original signal labels using in-training measurement scores; optimizer >= and final strict > remain distinct.',
            'Actual-length descriptor clipping and one-cycle FFT are declared generalizations; retain NaN handling and explicit rejection of undefined source cases.'],
        reuse_work=[
            'Inspect registered alternatives before adding generic baseline, directional peak, descriptor, Fourier and threshold operations.',
            'Expose numeric operation inputs and controls separately from three-signal grouping and fixed nine-feature recipe layout.',
            'Keep complete ensemble training and threshold fitting; do not substitute fixed probabilities or a reduced model count.',
            'Qualify original final-output scope and exact reusable bindings before stored graph promotion.'],
        limits=['Readiness only; historical hashes do not establish current numerical execution.',
            'Source notebook reuse permission remains unresolved; preserve the existing independent implementation and do not copy notebook code.',
            'Public algorithm-template descriptions only; no runtime signal records or identifying metadata retained.'],
        auditor_sha256=sha(__file__))


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/decomposed_vsb_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(original=result['original']['status'],reference_nodes=len(result['derivative_nodes']),
        checked=result['distinct_file_hash_checks'],drift=len(result['implementation_drift']),remaining=result['draft_originals_with_approved_derivatives'])))
