"""Qualify both sound_scale originals for explicitly bounded corrected reuse."""
import asyncio
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_remaining_legacy_physics_revisions import ROOT,sha,require
from scripts.check_sound_scale_identity_parent import check_parent
from scripts.validate_sound_scale_immutable_source import validate as validate_source
from scripts import validate_sound_scale_execution as oracle
from scripts.audit_legacy_physics_execution_scope import scope
from scripts.audit_physics_projection_lineage import signature,digest
from scripts.audit_physics_legacy_reconciliation import source_inventory
from sciona.physics_ingest.sound_scale_proof import SOURCE_VERSION


async def numerical(source):
    with patch.object(oracle,'validate_source',side_effect=validate_source):
        result=await oracle.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
    original=json.loads((ROOT/'docs/reviews/sound_scale_execution.json').read_text())
    normalized=json.loads(json.dumps(result));normalized['source_proof'].pop('immutable_validator_sha256')
    require(normalized==original,'Fresh immutable sound_scale execution differs')
    return result


def audit(source):
    execution=asyncio.run(numerical(source))
    baseline=json.loads((ROOT/'docs/reviews/physics_remaining_source_pairs.json').read_text())
    pairs=[g for g in baseline['graphs'] if g['derivation_ids']==['608598']]
    require(len(pairs)==1,'One sound_scale source pair required');pair=pairs[0]
    versions=[pair['legacy_version_id'],pair['projected_version_id']]
    require(versions[1]==SOURCE_VERSION,'Qualified projected source differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph,parent=check_parent(db);inventories=[];forms=[];sources=[]
        for version,key in zip(versions,['legacy','projected']):
            row=db.execute('SELECT a.artifact_id::text,v.version_id::text,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(version,)).fetchone()
            require(row and row['artifact_id']==pair[key+'_artifact_id'] and row['content_hash']==pair[key+'_content_hash'],'Original sound_scale identity changed')
            nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(version,)).fetchall()
            edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id',(version,)).fetchall()
            bindings=db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',(version,)).fetchall()
            forms.append(scope(nodes,edges));inventories.append(source_inventory([json.loads(n['type_signature']) for n in nodes],bindings));sources.append(row)
        a,b=forms
        require(signature(a[0])==signature(b[0]) and a[2]==b[2] and {k:Counter(v) for k,v in a[1].items()}=={k:Counter(v) for k,v in b[1].items()},'Complete sound_scale grouped scope differs')
        require(inventories[0]['relations']==inventories[1]['relations'] and inventories[0]['binding_versions']==inventories[1]['binding_versions']
            and all(i['all_bindings_active'] for i in inventories),'Sound scaling inference/binding contracts differ')
    reports=[]
    for key,row in zip(['legacy','projected'],sources):
        reports.append(dict(family='sound_scale_'+key,legacy_artifact_id=row['artifact_id'],legacy_version_id=row['version_id'],legacy_content_hash=row['content_hash'],
            projected_source_version_id=SOURCE_VERSION,approved_execution_version_id=parent['version_id'],approved_execution_graph_sha256=parent['graph_sha256'],
            qualification=parent,source_steps=len(a[0]),complete_grouped_scope_sha256=digest(signature(a[0])),complete_pair_multiplicity=True,
            complete_dependency_edges=True,new_atoms_required=0))
    return dict(read_only=True,approved=False,catalog_mutations=0,graphs=reports,fresh_execution=execution,
        auditor_sha256=sha(__file__),parent_checker_sha256=sha(ROOT/'scripts/check_sound_scale_identity_parent.py'),
        limitations=['Conditional Rydberg bonding-energy model and isotropic longitudinal elasticity; caller establishes material applicability.',
            'Dimensionless atomic mass A=m/mp >=1, positive constants and bulk prefactor, nonnegative shear-to-bulk ratio; identical nonempty shapes.',
            'Correct the missing factor two in the mass denominator and source symbol interpretation; retain omitted prefactor and shear contributions explicitly, without literal source parity.',
            'Reuse unchanged provider for scaling estimate, upper scale, bulk speed, longitudinal speed and shear factor. Upper scale bounds only the unit-prefactor bulk-only curve; diagnostic outputs may exceed it.',
            'Independent float64 rounding with local 450-digit arithmetic and positive representable outputs; no universal material-speed bound or empirical material certification.'])



if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-directory',type=Path,required=True)
    report=audit(p.parse_args().source_directory)
    (ROOT/'docs/reviews/sound_scale_identity_execution_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=False,graphs=len(report['graphs']),new_atoms=0)))
