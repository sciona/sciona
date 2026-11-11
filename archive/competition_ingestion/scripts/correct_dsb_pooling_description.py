"""Correct draft pooling prose while retaining the original metadata evidence."""
import json
from pathlib import Path
from uuid import UUID
from scripts.complete_dsb_draft_metadata import PLAN, ROOT, connect, protected, GENERIC_LIMIT
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score
from scripts.inventory_dsb_decomposed_providers import sha
from scripts.import_residual_execution_drafts import comparable

TEXT=('Combine probabilities by multiplying their complements with the complement of the '
      'converted baseline probability, then subtracting that product from one. '+GENERIC_LIMIT)
RUNNER='dsb-draft-pooling-description.v2'


def targets():
    frozen=json.loads(PLAN.read_text())
    intake=json.loads((ROOT/'docs/reviews/dsb_provider_intake_plan.json').read_text())
    identity=next(r['atom_id'] for r in intake['new_providers'] if r['runtime_fqdn'].endswith('.noisy_or_with_baseline'))
    rows=[r for r in frozen['rows'] if r['table'] in ('atom_descriptions','artifact_descriptions')
          and r['row']['kind']=='dejargonized' and identity in r['key'].values()]
    if len(rows)!=2:raise ValueError('Expected two draft descriptions')
    return frozen,rows


def change(apply=False,fail=False):
    frozen,rows=targets();writes=0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        if protected(db,list(map(UUID,frozen['identities'])))!=frozen['preserved']:
            raise ValueError('Publication or history changed')
        for item in rows:
            expected=item['row'];new=dict(expected,content=TEXT,generated_by=RUNNER,
                                         jargon_score=round(estimate_jargon_score(TEXT),6))
            predicate=' AND '.join(k+'=%s' for k in item['key'])
            args=tuple(item['key'].values())
            actual=comparable(db.execute('SELECT '+','.join(expected)+' FROM '+item['table']+' WHERE '+predicate+' FOR UPDATE',args).fetchall())
            if actual==[new]:continue
            if actual!=[expected]:raise ValueError('Description changed outside the reviewed correction')
            db.execute('UPDATE '+item['table']+' SET content=%s,generated_by=%s,jargon_score=%s WHERE '+predicate,
                       (TEXT,RUNNER,new['jargon_score'],*args))
            writes+=1
            if fail:raise RuntimeError('Injected correction rollback')
        if protected(db,list(map(UUID,frozen['identities'])))!=frozen['preserved']:
            raise ValueError('Correction changed publication or history')
        if not apply:db.rollback()
    return writes


if __name__=='__main__':
    dry=change()
    if dry!=2:raise ValueError('Expected original draft descriptions')
    try:change(fail=True)
    except RuntimeError as exc:
        if str(exc)!='Injected correction rollback':raise
    else:raise ValueError('Expected injected rollback')
    if change()!=2:raise ValueError('Rollback incomplete')
    applied=change(apply=True)
    repeated=change(apply=True)
    if applied!=2 or repeated!=0:raise ValueError('Correction was not idempotent')
    report=dict(approved=False,descriptions_corrected=applied,repeat_writes=repeated,rollback_verified=True,
                publication_and_history_preserved=True,reason='The product is over probability complements, not probabilities.',
                replacement=TEXT,prior_metadata_plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)))
    (ROOT/'docs/reviews/dsb_pooling_description_correction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
