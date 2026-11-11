"""Reconstruct and map the exact original ARC intake to the reviewed full solver."""
import hashlib
import json
import subprocess
from pathlib import Path
from sciona.competition_import import FORMAT, _canonical, _id
from scripts.import_flavours_provider_drafts import ROOT
from scripts.promote_arc_execution import SOURCE_VERSION, SOURCE_HASH, sha
from scripts.review_arc_provider_contracts import review as provider_review
from scripts.verify_arc_metadata import verify


def review():
    contracts = provider_review()
    verify()
    source = ROOT.parent/'sciona-atoms/data/solution_cdgs'
    name = 'arc_program_synthesis_1st'
    snapshot = dict(format=FORMAT, template=json.loads((source/(name+'.json')).read_text()),
                    bindings=json.loads((source/(name+'_bindings.json')).read_text()))
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
    if digest != SOURCE_HASH or _id(fqdn+'/'+digest) != SOURCE_VERSION:
        raise ValueError('Original snapshot reconstruction differs')
    path = 'docs/reviews/competition_arc_source_contract_review.json'
    raw = subprocess.check_output(['git','show','7aa2448a:'+path],cwd=ROOT)
    if raw != (ROOT/path).read_bytes():
        raise ValueError('Reviewed source corrections changed')
    corrections = {r['stage']:r for r in json.loads(raw)['corrections']}
    stages = snapshot['template']['stages']
    if len(stages)!=5 or {s['stage_id'] for s in stages}!=set(corrections):
        raise ValueError('Exact five-stage coverage required')
    pins = json.loads((ROOT/'docs/reviews/competition_arc_source_pins.json').read_text())
    hashes = {f['path']:f['sha256'] for f in pins['files']}
    return dict(format='arc-original-scope.v1',passed=True,approved=False,catalog_mutations=0,
        artifact_id=_id(fqdn),fqdn=fqdn,source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
        source_snapshot_reconstructed=True,source_commit=pins['commit'],
        source_stage_review=[dict(stage_id=s['stage_id'],original_description=s['description'],
            realization=corrections[s['stage_id']]['actual'],
            software_sources=[dict(path=p,sha256=hashes[p]) for p in corrections[s['stage_id']]['source']]) for s in stages],
        reuse_decision=contracts['reuse_decision'],limitations=contracts['limitations'],
        output_scope='Ranked grid predictions and search status from the complete four-phase C++ solver. No saved-program, perfect training agreement, Python A* or competition submission-format claim.',
        provider_contracts_sha256=sha(ROOT/'docs/reviews/arc_provider_contracts.json'),
        source_review_sha256=sha(ROOT/path),reviewer_sha256=sha(Path(__file__)),
        remaining=['Corrected catalog graph execution and immutable evidence.',
                   'Staging, rollback qualification, Tier 3 publication and production selection.'])


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/arc_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,stages=len(report['source_stage_review']),approved=False)))
