"""Review frozen transductive feature replay for concrete Tier 3 use."""
import json
import platform

import numpy as np
import pandas as pd
from scripts.import_osic_feature_replay_drafts import reviewed_plan,ROOT,PLAN,sha
from scripts.import_flavours_provider_drafts import connect
from scripts.validate_osic_feature_replay_provider import validate

LIMITS=[
    'Automated Tier 3 concrete in-process frozen preprocessing only; abstract execution rejects. No human Tier 1 or empirical Tier 2 claim.',
    'Exact declared transductive query population and record order are required. A changed population requires separately qualified preparation and model training, not silent reuse of this state.',
    'Caller supplies private numeric preparation state, mapped entity/time queries and matching feature/response coordinate labels. Baseline observations and query scope must remain private runtime material.',
    'State validation is structural, not authentication. Caller establishes provenance, unambiguous column names, category meanings, coordinate units, numeric representability and actual observation availability.',
    'Qualification covers float32 and float64 saved arithmetic. Integer arithmetic options are structurally supported but not comprehensively qualified; overflow, underflow and lossy casting require caller validation.',
    'The stacked feature matrix can have a promoted dtype from categorical columns. Model-input casting, structured column serialization, model replay, prediction reduction and branch blending are separate operations.',
    'Fresh synthetic replay reproduces qualified fitted feature preparation, availability masks and observed values without fitting. This is not an independent source-notebook comparison or empirical transfer evidence.',
    'Source-specific preprocessing is separated from reusable downstream model execution and numerical operations. No unrestricted train-only scaler, HTTP transport, concurrent-mutation safety or original-CDG approval claim.',
]


def review():
    intake=reviewed_plan()
    paths={'provider':ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/osic_feature_replay.py',
           'registry':ROOT/'sciona/ghost/registry.py',
           'implementation':ROOT/'sciona/osic_feature_state.py',
           'feature_software':ROOT/'sciona/osic_quantile_features.py'}
    for name in ('review_osic_feature_replay_semantics','import_osic_feature_replay_drafts',
                 'plan_osic_feature_replay_intake','validate_osic_feature_replay_provider'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    execution=validate()
    if execution!=json.loads((ROOT/'docs/reviews/osic_feature_replay_provider.json').read_text()):
        raise ValueError('Fresh feature replay qualification differs')
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
                plain_description='Use saved preparation rules to turn the same declared query rows into model features and mark which rows have known values.',
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Separate frozen source preprocessing from model-input conversion, model replay and reduction. Cross-domain use requires an explicit compatible mapping, coordinate and availability contract; fixed transductive query scope cannot be generalized away.',
                registered_provider_dependencies=[]))
    if before!={name:sha(path) for name,path in paths.items()} or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='osic-feature-replay-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,software_sha256=before,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1,foreground_cpu_budget=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/osic_feature_replay_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),approved=False)))
