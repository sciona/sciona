"""Review pure container routing for concrete Tier 3 use."""
import json
import platform

import numpy as np
from scripts.import_ordered_fields_drafts import reviewed_plan,ROOT,PLAN,sha
from scripts.import_flavours_provider_drafts import connect
from scripts.validate_ordered_fields_provider import validate

LIMITS=[
    'Automated Tier 3 concrete in-process routing only; abstract execution rejects. No human Tier 1 or empirical Tier 2 claim.',
    'Caller supplies a plain dict and explicit unique nonblank string field names in a plain list or tuple. No inferred schema, implicit name trimming or default missing values.',
    'The result is a new outer list of borrowed values. Values are neither inspected nor copied, validated or converted. Existing aliases remain aliases; downstream value mutation affects caller-owned objects.',
    'Caller establishes field meanings, dtype and shape contracts, row alignment, units, observation availability and downstream model compatibility.',
    'Synthetic motion and electrical-power examples establish explicit routing reuse, not empirical effectiveness. Python container behavior is the implementation; NumPy is used only by numerical fixtures.',
    'The pinned image-slice graph is supplementary prior composition evidence. This review reruns the routing qualification, not image inference or full CDG training.',
    'No HTTP transport, serialization, thread-safety under concurrent mutation or original-CDG approval is established.',
]


def review():
    intake=reviewed_plan()
    paths={'provider':ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/pipeline/ordered_fields.py',
           'registry':ROOT/'sciona/ghost/registry.py'}
    for name in ('review_ordered_fields_semantics','import_ordered_fields_drafts',
                 'plan_ordered_fields_intake','validate_ordered_fields_provider'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    execution=validate()
    if execution!=json.loads((ROOT/'docs/reviews/ordered_fields_provider.json').read_text()):
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
                plain_description='Select named values from the supplied mapping and return them in the requested order.',
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Application-independent explicit field routing; synthetic mechanical-motion and electrical-power examples preserve order, values and aliasing. Schema, units and downstream interpretation remain caller-owned.',
                registered_provider_dependencies=[]))
    if before!={name:sha(path) for name,path in paths.items()} or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='ordered-fields-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,software_sha256=before,
        runtime=dict(python=platform.python_version(),numpy_fixture_version=np.__version__),
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1,foreground_cpu_budget=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/ordered_fields_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),approved=False)))
