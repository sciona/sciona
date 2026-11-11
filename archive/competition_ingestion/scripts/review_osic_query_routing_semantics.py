"""Review pure container routing for concrete Tier 3 use."""
import json
import platform
import numpy as np
import pandas as pd

from scripts.import_osic_query_routing_drafts import reviewed_plan,ROOT,PLAN,sha
from scripts.import_flavours_provider_drafts import connect
from scripts.validate_osic_query_routing import validate

LIMITS=[
    'Automated Tier 3 concrete in-process query routing only; no human Tier 1 or empirical Tier 2 claim.',
    'Exact nonempty plain DataFrame with key/entity/time columns, unique nonblank string keys, nonblank string entities and native int64 times. Reject nullable integers and floating times.',
    'Copied entity/time rows and supplied keys retain original order, index, exact strings, repeated coordinates and integer dtype. Output cell mutations do not affect input cells.',
    'Caller owns key meanings, coordinate units, observation availability and exact frozen query-scope agreement. No sorting, generation, parsing, fitting, merging or numerical conversion.',
    'Source-schema boundary only; reusable feature transformations, model replay and numerical reductions stay separate. No empirical effectiveness claim.',
    'Six column permutations, integer extremes and eighteen rejection cases are freshly checked. The pinned seven-node keyed graph is prior composition evidence; training is not rerun by this review.',
    'No HTTP transport, serialization, pandas index internals, concurrent mutation safety or original-CDG approval is established.',
]


def review():
    intake=reviewed_plan()
    paths={'provider':ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/osic_query_routing.py',
           'registry':ROOT/'sciona/ghost/registry.py'}
    for name in ('review_osic_query_routing_semantics','import_osic_query_routing_drafts',
                 'plan_osic_query_routing_intake','validate_osic_query_routing'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    execution=validate()
    if execution!=json.loads((ROOT/'docs/reviews/osic_query_routing.json').read_text()):
        raise ValueError('Fresh routing qualification differs')
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for item in intake['new_providers']:
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                rows=db.execute(f'''SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                    WHERE a.{key}=%s AND v.is_latest''',(item['atom_id'],)).fetchall()
                if rows!=[dict(fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                               content_hash=item['content_hash'],trust_tier=3)]:
                    raise ValueError('Exact current provider version required')
                ports=db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(item['version_id'],)).fetchall()
                if ports!=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])):
                    raise ValueError('Stored interface differs')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(item['version_id'],)).fetchone():
                raise ValueError('No registered atom dependencies expected')
            records.append(dict(artifact_id=item['atom_id'],fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                content_hash=item['content_hash'],interfaces=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])),
                operation=item['seed_fields']['description'],
                plain_description='Copy query rows and return their supplied keys in the same order.',
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Source-schema query boundary that separates key alignment from reusable numerical operations and model replay. Identity meanings and units remain caller-owned.',
                registered_provider_dependencies=[]))
    if before!={name:sha(path) for name,path in paths.items()} or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='osic-query-routing-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,software_sha256=before,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1,foreground_cpu_budget=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/osic_query_routing_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),approved=False)))
