"""Explain the exact omitted malformed source step; do not approve either original."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.audit_legacy_physics_execution_scope import scope, require
from scripts.audit_physics_projection_lineage import ROOT, signature, digest
from scripts.audit_physics_legacy_reconciliation import source_inventory
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts.validate_wave_interference_immutable_source import validate as validate_source
from scripts import validate_wave_interference_execution as oracle

STEP=('539398','6988426')
RELATION='pdg_remote:539398:6988426:7607271250:4192519596'
EXTRA_BINDING=('physics.pdg.remote_wave.equation.7607271250',
               '2d737e879e5f8d7d9485b619c2f1443b7c32266b2a24a9913be9e29640eded20')
EXTRA_EDGES={(STEP,('539398','1791333')),(STEP,('539398','3264098'))}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def numerical(source):
    with patch.object(oracle,'validate_source',side_effect=validate_source):
        result=await oracle.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
    normalized=json.loads(json.dumps(result));normalized['source_proof'].pop('immutable_validator_sha256')
    require(normalized==json.loads((ROOT/'docs/reviews/wave_interference_execution.json').read_text()),
            'Fresh corrected execution differs from retained qualification')
    return result


def reconcile(source):
    execution=asyncio.run(numerical(source))
    pins=execution['source_proof']['source_file_sha256']
    require(sha(source/'steps.cypher')==pins['conversion_of_data_formats/steps.cypher'],'Step source pin differs')
    links=[]
    for block in re.split(r'(?m)^UNWIND ',(source/'steps.cypher').read_text()):
        if not block.startswith('[{start: {id:"6988426"}'):
            continue
        target=re.search(r'end: \{id:"(\d+)"',block)
        kind=re.search(r'\[r:(\w+)\]',block)
        require(target is not None and kind is not None,'Unrecognized pinned source relationship')
        links.append((kind[1],target[1]))
    require(links==[('HAS_INFERENCE_RULE','111984'),('HAS_INPUT','7607271250'),
                    ('HAS_FEED','4182362050'),('HAS_FEED','1742775076'),('HAS_FEED','4583868070'),
                    ('HAS_OUTPUT','4192519596')],'Pinned malformed step changed')
    pairs=[g for g in json.loads((ROOT/'docs/reviews/physics_remaining_source_pairs.json').read_text())['graphs']
           if g['derivation_ids']==['539398']]
    require(len(pairs)==1,'Unique wave-interference source pair required');pair=pairs[0]
    forms=[]; inventories=[]; originals=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for key in ('legacy','projected'):
            version=pair[key+'_version_id']
            row=db.execute('SELECT artifact_id::text,version_id::text,content_hash FROM artifact_versions WHERE version_id=%s',
                           (version,)).fetchone()
            require(row==dict(artifact_id=pair[key+'_artifact_id'],version_id=version,content_hash=pair[key+'_content_hash']),
                    'Immutable source identity differs')
            nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s',(version,)).fetchall()
            edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s',(version,)).fetchall()
            bindings=db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',
                                (version,)).fetchall()
            forms.append(scope(nodes,edges))
            inventories.append(source_inventory([json.loads(n['type_signature']) for n in nodes],bindings))
            originals.append(dict(kind=key,**row,nodes=len(nodes),steps=len(forms[-1][0]),bindings=len(bindings),
                                  original_history_sha256=history(db,version)))
    legacy,projected=forms; a,b=inventories
    require(len(legacy[0])==26 and len(projected[0])==27 and set(projected[0])-set(legacy[0])=={STEP}
            and not set(legacy[0])-set(projected[0]),'Unexpected step-set difference')
    common={k:v for k,v in projected[0].items() if k!=STEP}
    require(signature(legacy[0])==signature(common),'Shared grouped step contracts differ')
    require(all(Counter(v)==Counter(projected[1][k]) for k,v in legacy[1].items()),'Shared relation multiplicity differs')
    require(set(projected[2])-set(legacy[2])==EXTRA_EDGES and not set(legacy[2])-set(projected[2]),
            'Unexpected dependency-edge difference')
    require(a['all_bindings_active'] and b['all_bindings_active'],'Inactive original binding')
    require(set(b['relations'])-set(a['relations'])=={RELATION} and not set(a['relations'])-set(b['relations'])
            and all(v==b['relations'][k] for k,v in a['relations'].items()),'Unexpected inference-contract difference')
    require(set(b['binding_versions'])-set(a['binding_versions'])=={EXTRA_BINDING}
            and not set(a['binding_versions'])-set(b['binding_versions']),'Unexpected bound-version difference')
    missing=projected[0][STEP]
    require(missing['rule']=='111984' and missing['bindings']['feeds']==[
        dict(feed_id='1742775076',latex='Z',sympy="Symbol('pdg0003192')"),
        dict(feed_id='4583868070',latex='B',sympy="Symbol('pdg0004698')")], 'Omitted step rule or feeds differ')
    require(len(projected[1][STEP])==1,'Omitted step pair multiplicity differs')
    corrections=execution['source_proof']['corrections']
    require('Step12 binds a theta feed instead of the polar-Z expression; restore the polar expression and both variable substitutions.'
            in corrections,'Existing correction does not cover the omitted malformed step')
    result=dict(passed=True,read_only=True,approved=False,eligible_for_approval=False,catalog_mutations=0,
        originals=originals,shared_source_steps=26,shared_pair_multiplicity_verified=True,
        shared_scope_sha256=digest(signature(legacy[0])),omitted_step=STEP,projected_only_relation=RELATION,
        projected_only_binding=EXTRA_BINDING,projected_only_edges=sorted(EXTRA_EDGES),pinned_step_links=links,
        source_step_file_sha256=sha(source/'steps.cypher'),fresh_execution=execution,
        reconciliation='The legacy representation omits the one malformed input/feed step; all other grouped contracts and pair multiplicities agree. The projected representation preserves that malformed step and its two downstream dependencies. Both require the already documented corrected polar-amplitude reconstruction; neither original graph is certified as a valid proof.',
        required_before_promotion=['Verify the approved corrected parent and provider, including exact bindings, ports, evidence and physical validity bounds.',
            'Stage separate corrected versions under both original identities with this explicit omission/correction provenance.',
            'Qualify stored execution, corruption rejection, rollback, publication idempotence and fresh served selection/execution.'],
        reconciler_sha256=sha(__file__),immutable_validator_sha256=sha(ROOT/'scripts/validate_wave_interference_immutable_source.py'))
    return json.loads(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=reconcile(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/wave_interference_original_reconciliation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=False,shared_steps=26,omitted_steps=1,catalog_mutations=0)))
