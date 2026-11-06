"""Qualify the corrected source interpretation without rewriting historic evidence.

Historical source probes remain historical. Current implementation qualification
comes from the full native execution review; stored-original execution is a
separate publication requirement.
"""
import json
import subprocess
from scripts.review_openvaccine_execution import SOURCES, COMMITS, LIMITATIONS
from scripts.review_decomposed_openvaccine_implementation import review as implementation_review
from scripts.stage_openvaccine_corrected_intakes import ROOT, TARGETS, CORRECTIONS, SCOPE, check_sources, connect, sha

SOURCE_COMMIT='fe31a928384c0a4aa483c1cb7df4951a7c6ad017'
REPORT=ROOT/'docs/reviews/openvaccine_corrected_source_review.json'


def immutable(path):
    relative=str(path.relative_to(ROOT))
    content=subprocess.check_output(['git','show',SOURCE_COMMIT+':'+relative],cwd=ROOT,stderr=subprocess.PIPE)
    if content!=path.read_bytes():
        raise ValueError('Immutable source evidence changed: '+relative)
    return sha(path)


def review():
    path=ROOT/'docs/reviews/competition_openvaccine_semantic_review.json'
    immutable(path)
    historical=json.loads(path.read_text())
    if (historical['source_versions']!=SOURCES or historical['source_commits']!=COMMITS
            or historical['limitations']!=LIMITATIONS or historical['verdict']!='acceptable_with_limits'
            or historical['proposed_tier']!=3 or historical['review_source']!='automated'):
        raise ValueError('Historical source qualification differs')
    evidence={str(path.relative_to(ROOT)):sha(path)}
    for field,base in [('evidence_sha256',ROOT/'docs/reviews'),('auxiliary_sha256',ROOT)]:
        for name,digest in historical[field].items():
            item=(base/name).resolve()
            if not item.is_relative_to(base.resolve()) or immutable(item)!=digest:
                raise ValueError('Historical evidence or source changed')
            evidence[str(item.relative_to(ROOT))]=digest
    for name in ('source_triage','original_notebook_review','dependency_review'):
        item=ROOT/('docs/reviews/competition_openvaccine_'+name+'.json')
        evidence[str(item.relative_to(ROOT))]=immutable(item)
    qualified=implementation_review()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check_sources(db)
    # The first historical limitation says originals remain draft. Preserve it
    # in the old report; replace that state-dependent statement in this review.
    limits=[
        'Automated Tier 3 corrected reconstruction. Original identities and immutable histories are preserved. Tier 1 human certification and Tier 2 usage qualification are not claimed.',
        *LIMITATIONS[1:],
        'The two intake identities describe the same solution family and share the exact same six providers. They are not independent algorithms or separate replication evidence.',
        'The registered ordered numeric sum is domain independent. Feature-level training can accept compatible caller-defined graph features within fixed 55/8/5 dimensions and pinned architectures; cross-domain predictive quality is unproven. RNA folding and ensemble/training policy remain recipe-specific.',
        'Historical source probes are retained without claiming they were rerun. Current native implementation evidence supersedes the historical environment qualification; each stored original additionally requires full execution and fresh reference comparison.',
    ]
    return dict(format='openvaccine-corrected-source-review.v1',review_source='automated',
        proposed_tier=3,verdict='acceptable_with_limits',catalog_mutations=0,
        original_publication_approved=False,source_evidence_commit=SOURCE_COMMIT,
        source_commits=COMMITS,source_versions=SOURCES,targets=TARGETS,
        source_corrections=CORRECTIONS,scope=SCOPE,limitations=limits,
        source_evidence_sha256=evidence,implementation_review=qualified,
        reviewer_sha256=sha(__file__),intake_builder_sha256=sha(ROOT/'scripts/stage_openvaccine_corrected_intakes.py'))


if __name__=='__main__':
    result=review()
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3,originals=2,
        shared_solution_families=1,historical_evidence_files=len(result['source_evidence_sha256']),catalog_mutations=0)))
