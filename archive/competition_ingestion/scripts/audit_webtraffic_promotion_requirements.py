"""Review Community promotion requirements without imposing Tier1 certification gates."""
import hashlib
import json
from pathlib import Path


def audit(root):
    reports={};entries=0
    for path in sorted((root/'docs/reviews').glob('competition_webtraffic_*.json')):
        if path.name=='competition_webtraffic_promotion_requirements.json':continue
        report=json.loads(path.read_text())
        hashes=report.get('implementation_sha256',{})
        if not hashes:continue
        for name,expected in hashes.items():
            target=root/name
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('Stale evidence: '+name)
            entries+=1
        reports[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    association=root.parent/'sciona/association.pdf'
    expected_artifacts={
        'provider':root.parent/'sciona-atoms-dl/src/sciona/atoms/dl/webtraffic_execution.py',
        'graph':root/'sciona/webtraffic_graph.py',
        'publication_script':root/'scripts/promote_webtraffic_execution.py',
        'served_execution_verifier':root/'scripts/verify_webtraffic_publication.py'}
    return dict(approved=False,proposed_tier=3,
        policy_source=dict(path='../sciona/association.pdf',sha256=hashlib.sha256(association.read_bytes()).hexdigest()),
        policy_interpretation={
            'human_review_required_for_tier3':False,
            'tier2':'Rigorous adversarial testing plus community usage; not established by present evidence.',
            'tier1':'Independent expert certification; not requested for this promotion.',
            'not_blanket_tier3_gates':['Exact TensorFlow random sequence or GPU kernel replay',
                                      'Executing every competition-length training step',
                                      'Original TensorFlow checkpoint file interoperability'],
            'still_required':'Accurate bounded semantics, executable stage/boundary closure, version-bound automated review, provenance/license and canonical served execution.'},
        evidence=dict(reports=len(reports),implementation_hash_entries=entries,report_sha256=reports,
                      demonstrated=['Feature preparation and batching','Source-width batch256 three-model gradient/update execution',
                                    'Adam and EMA arithmetic, explicit stochastic inputs and observed EMA snapshots',
                                    'Saved-state continuation and ensemble prediction']),
        proposed_contract_review={
            'status':'required_before_approval',
            'preserve':['Original intake stays immutable/draft; derived version explicitly identifies corrected GRU/s32 architecture',
                        'All training/prediction stages executable, including reproducible initialization and checkpoint/ensemble state',
                        'Explicit runtime stochastic inputs and EMA observation behavior must be part of public callable contract'],
            'exclude_claims':['Original source seeded-run/competition-score reproduction','Tier1 certification or Tier2 community-usage qualification'],
            'disconnected_attention':'s32 does not consume attention; omission requires explicit semantic review and prevents original full-checkpoint equivalence claims.'},
        artifacts_present={k:p.is_file() for k,p in expected_artifacts.items()},
        remaining=['Finalize and automatically review complete executable boundary semantics for randomness, EMA observations and checkpoint representation',
                   'Register provider functions and shape witnesses; build immutable derived execution graph',
                   'Exercise serialized graph with positive/negative synthetic lifecycle cases and hash-bound dependency closure',
                   'Apply idempotent Tier3 publication transaction and verify served retrieval, materialization and execution'],
        catalog_status='No mutation; last separately verified snapshot remains72served CDGs,142competition intakes,5covered by canonical served derivatives.')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];report=audit(root)
    (root/'docs/reviews/competition_webtraffic_promotion_requirements.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(proposed_tier=report['proposed_tier'],approved=False,
                         reports=report['evidence']['reports'],hash_entries=report['evidence']['implementation_hash_entries'],
                         artifacts_present=report['artifacts_present'],remaining=report['remaining'])))
