"""Audit the next original against live catalog and retained synthetic evidence."""
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from scripts.import_flavours_provider_drafts import ROOT,connect
from sciona.competition_import import _canonical

ORIGINAL='a6298d16-4fa7-5577-b559-7c17c006ccd5'
IDENTITY='7217a6e8-5fcf-58a0-b0b1-caad0e40a20a'
SOURCE_HASH='dc85ebd516bf6b3832bcab2bf4262e9603b7c8ef2cf2317215495182aef88b8d'
DERIVATIVE='41712163-61bd-5345-b735-bca98e5c42b2'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    reviews=ROOT/'docs/reviews'
    evidence={};checked={};drift=[]
    for path in sorted(reviews.glob('competition_webtraffic_*.json')):
        report=json.loads(path.read_text())
        evidence[path.name]=sha(path)
        for relative,digest in report.get('implementation_sha256',{}).items():
            key=(relative,digest)
            if key in checked:continue
            current=sha(ROOT/relative) if (ROOT/relative).is_file() else None
            checked[key]=current
            if current!=digest:drift.append(dict(file=relative,recorded_sha256=digest,current_sha256=current))
    inventory={}
    for path in sorted((ROOT/'sciona').glob('webtraffic_*.py')):
        tree=ast.parse(path.read_text())
        inventory[path.name]=dict(sha256=sha(path),functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))])
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s''',(ORIGINAL,)).fetchone()
        if not original or original['artifact_id']!=IDENTITY or original['content_hash']!=SOURCE_HASH:
            raise ValueError('Original identity drift')
        source=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
        if len(source)!=1 or hashlib.sha256(_canonical(source[0]['details']['snapshot']).encode()).hexdigest()!=SOURCE_HASH:
            raise ValueError('Exact original source snapshot required')
        template=source[0]['details']['snapshot']['template']
        stages=[dict(stage_id=s['stage_id'],description=s['description']) for s in template['stages']]
        derivative=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s''',(DERIVATIVE,)).fetchone()
        if not derivative or derivative['status']!='approved' or not derivative['is_publishable'] or not derivative['is_latest']:
            raise ValueError('Approved derivative required')
        nodes=db.execute('SELECT node_id FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(DERIVATIVE,)).fetchall()
        bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(DERIVATIVE,)).fetchall()
        served=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s',(derivative['artifact_id'],)).fetchone()['n']
        if served!=1 or len(nodes)!=3 or len(bindings)!=3:raise ValueError('Complete served three-stage derivative required')
    return dict(format='decomposed-webtraffic-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        original=original,source_version_id=ORIGINAL,source_stages=stages,derivative=derivative,
        derivative_version_id=DERIVATIVE,derivative_nodes=nodes,derivative_bindings=bindings,
        retained_evidence_sha256=evidence,distinct_file_hash_checks=len(checked),implementation_drift=drift,
        implementation_inventory=inventory,current_runtime_versions={name:metadata.version(name) for name in ('numpy','pandas','torch')},
        source_corrections=[
            'Retained pinned-source review identifies GRU rather than intake LSTM; active s32 width267,283 input days and63 forecast days rather than fixed60/60.',
            'Missing values and lag masks drive replacement with zero; do not infer explicit mask-feature concatenation.',
            'Preserve complete feature/window preparation, three-model training, Adam, EMA observation phases, checkpoint cadence, retention and count-space ensemble.',
            'CPU NumPy/PyTorch random streams and explicit EMA phases are declared adaptations; no identical-seed legacy TensorFlow or source race replay.',
            'Competition-specific date slicing and submission routing are excluded by the derivative. Original final-output scope must be resolved explicitly before promotion.'],
        reuse_work=[
            'Inventory registered alternatives for numeric preparation, correlation, normalization, recurrent cells, optimizer updates, EMA and ensemble before adding providers.',
            'Expose numerical blocks with independent numeric inputs and controls; retain page parsing and source feature order as named recipe policies.',
            'Preserve recurrent state, gradient flow, every configured update and all retained checkpoints when decomposing lifecycle orchestration; do not replace complete training with a forward-only adapter.'],
        next_actions=[
            'Verify pinned public software against source corrections and resolve original final-output scope.',
            'Run the unchanged complete synthetic derivative under the current runtime with serial workers and one-thread pools; retain old evidence.',
            'Qualify reusable decomposition, exact registered bindings and stored original execution before automated Tier3 publication.'],
        limits=['Readiness audit only; matching file hashes do not rerun numerical evidence or establish current runtime equivalence.',
                'Source stage descriptions are public algorithm-template metadata, not runtime dataset contents.',
                'No original, derivative or provider rows changed.'],auditor_sha256=sha(__file__))


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/decomposed_webtraffic_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(original_status=result['original']['status'],derivative_nodes=len(result['derivative_nodes']),
        checked=result['distinct_file_hash_checks'],drift=result['implementation_drift'],runtime=result['current_runtime_versions'],catalog_mutations=0)))
