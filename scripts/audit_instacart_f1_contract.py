"""Prove the fixed-count scorer is not an independent-Bernoulli F1 optimizer."""
from fractions import Fraction
import hashlib
import inspect
from itertools import product
import json
from pathlib import Path

import numpy as np
from sciona.atoms.scoring.atoms import expected_f1_threshold
from scripts.import_flavours_provider_drafts import connect

ROOT=Path(__file__).resolve().parents[1]
FQDN='cdg.competition.solution.kaggle.instacart_basket_1st'


def expectation(probabilities,prediction):
    total=Fraction(0)
    for outcome in product((0,1),repeat=len(probabilities)):
        mass=Fraction(1)
        for probability,label in zip(probabilities,outcome):
            mass*=probability if label else 1-probability
        denominator=sum(prediction)+sum(outcome)
        score=Fraction(2*sum(a*b for a,b in zip(prediction,outcome)),denominator) if denominator else Fraction(0)
        total+=mass*score
    return total


def audit():
    probabilities=[Fraction(1,10),Fraction(3,10),Fraction(6,10)]
    assert sum(probabilities)==1
    candidate,_=expected_f1_threshold(np.array(list(map(float,probabilities))),1)
    candidate=tuple(map(int,candidate))
    alternatives=list(product((0,1),repeat=3))
    scores=[expectation(probabilities,prediction) for prediction in alternatives]
    best=alternatives[scores.index(max(scores))]
    actual=expectation(probabilities,candidate)
    assert candidate==(0,0,1) and best==(0,1,1)
    assert actual==Fraction(523,1000) and max(scores)==Fraction(2637,5000)
    assert actual<max(scores)
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        original=db.execute('''SELECT a.artifact_id::text,a.status,a.is_publishable,v.version_id::text,v.content_hash
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn=%s AND v.is_latest''',(FQDN,)).fetchone()
        providers=db.execute('''SELECT a.atom_id::text,a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash
            FROM atoms a JOIN atom_versions v USING(atom_id)
            WHERE a.source_symbol=%s AND v.is_latest''',('expected_f1_threshold',)).fetchall()
    path=Path(inspect.getfile(inspect.unwrap(expected_f1_threshold)))
    return dict(format='instacart-f1-contract-audit.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,fqdn=FQDN,original=original,related_provider_records=providers,
        counterexample=dict(probabilities=list(map(str,probabilities)),supplied_positive_count=1,
            supplied_count_equals_expected_cardinality=True,candidate_selection=list(candidate),optimal_selection=list(best),
            candidate_expected_f1=str(actual),optimal_expected_f1=str(max(scores)),
            outcome_states=8,prediction_sets=8,arithmetic='exact rational enumeration'),
        conclusion='The helper optimizes a fixed-count denominator objective; it cannot establish the intake independent-Bernoulli expected-F1 guarantee.',
        empty_set_policy='F1(empty,empty)=0. Both compared selections are nonempty, so their strict ordering is independent of this convention.',
        provider_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        required_next_steps=['Resolve first-place source identity and permission; the intake code URL was not retrievable in this review.',
            'Qualify the exact expected-F1 decision rule, including empty-outcome policy and deterministic ties, against independent enumeration.',
            'Keep probability estimation, grouping and domain output conventions separate from reusable decision computation.',
            'Reconstruct and execute the complete source workflow before original-intake promotion.'],
        limitations=['This does not refute the helper under a genuinely fixed known true cardinality; that is a different probability model.',
            'No claim is made that this synthetic counterexample reproduces the inaccessible first-place source.',
            'No provider or CDG status was changed by this audit.'])


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/instacart_f1_contract_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(counterexample_confirmed=True,candidate_expected_f1=result['counterexample']['candidate_expected_f1'],
        optimal_expected_f1=result['counterexample']['optimal_expected_f1'],catalog_mutations=0)))
