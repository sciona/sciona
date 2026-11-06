"""Resolve original source corrections and output boundary without catalog mutation."""
import hashlib
import json
import subprocess
from scripts.audit_decomposed_vsb_readiness import ROOT,sha,ORIGINAL,SOURCE_HASH
from scripts.import_flavours_provider_drafts import connect
from sciona.competition_import import _canonical

PIN='1c44c993'
PATHS=['docs/reviews/competition_vsb_execution_source_review.json',
       'docs/reviews/competition_vsb_semantic_review.json','docs/reviews/vsb_current_baseline.json']


def review():
    frozen={}
    for path in PATHS:
        raw=(ROOT/path).read_bytes()
        if raw!=subprocess.check_output(['git','show',PIN+':'+path],cwd=ROOT):raise ValueError('Immutable source/baseline review changed')
        frozen[path]=json.loads(raw)
    source=frozen[PATHS[0]];baseline=frozen[PATHS[2]]
    if source['source_version_id']!=ORIGINAL or source['source_content_hash']!=SOURCE_HASH or not baseline['passed']:
        raise ValueError('Qualified source baseline required')
    for name,digest in baseline['graph']['code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Reference software changed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
        if len(rows)!=1 or hashlib.sha256(_canonical(rows[0]['details']['snapshot']).encode()).hexdigest()!=SOURCE_HASH:
            raise ValueError('Immutable original snapshot required')
        template=rows[0]['details']['snapshot']['template']
        stages=[s['stage_id'] for s in template['stages']]
        if set(stages)!=set(source['stage_mapping']):raise ValueError('All original stages require explicit disposition')
        if template['outputs']!=[dict(name='fault_probability',required=True,type_desc='float',constraints='Binary fault probability in [0, 1].')]:
            raise ValueError('Original output contract changed')
    return dict(format='vsb-original-scope.v1',passed=True,approved=False,catalog_mutations=0,
        source_version_id=ORIGINAL,source_content_hash=SOURCE_HASH,original_stages=stages,
        source_corrections=source['stage_mapping'],original_outputs=template['outputs'],
        decision='Preserve full independent source reconstruction: raw signal triples through baseline removal, custom peaks, descriptors and phase aggregates; all 125 repeated-fold tree models; equal query probability averaging; signal-label MCC threshold and strict decisions broadcast over triples.',
        output_boundary='Return one probability per query measurement in input order, three ordered signal decisions per measurement, selected threshold, model count and declared evaluation limitations. The intake scalar shorthand is corrected to this batched numerical result. No file or submission artifact is required by the original root output.',
        reuse='Numeric filtering, peak ranking and knee controls, descriptor and phase operations, generic-width fold fitting and threshold search can be selected separately. Consecutive three-signal grouping, nine-feature permutation, random five-way fold coverage and threshold routing remain explicit recipe assumptions.',
        evidence_sha256={p:sha(ROOT/p) for p in PATHS},reviewer_sha256=sha(__file__),
        limits=['Automated scope resolution only; decomposed full graph execution and original publication remain pending.',
                'Retain independent mathematics and source attributions; no notebook code copying or new license-grant claim.',
                'Both heldouts influence stopping and threshold selection uses in-training scores; no unbiased validation or historical performance claim.',
                'Actual-length descriptors and FFT phase remain declared generalizations; preserve undefined-case rejection and missing aggregates.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/vsb_original_scope.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,original_stages=len(result['original_stages']),approved=False)))
