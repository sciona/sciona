"""Qualify both sine_double_angle originals for explicitly bounded corrected reuse."""
import asyncio
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_remaining_legacy_physics_revisions import ROOT,sha,require
from scripts.check_sine_double_angle_identity_parent import check_parent
from scripts.validate_sine_double_angle_immutable_source import validate as validate_source
from scripts import validate_sine_double_angle_execution as oracle
from scripts.audit_legacy_physics_execution_scope import scope
from scripts.audit_physics_projection_lineage import signature,digest
from scripts.audit_physics_legacy_reconciliation import source_inventory
from sciona.physics_ingest.sine_double_angle_proof import SOURCE_VERSION


async def numerical(source):
    with patch.object(oracle,'validate_source',side_effect=validate_source):
        result=await oracle.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
    original=json.loads((ROOT/'docs/reviews/sine_double_angle_execution.json').read_text())
    normalized=json.loads(json.dumps(result));normalized['source_proof'].pop('immutable_validator_sha256')
    require(normalized==original,'Fresh immutable sine_double_angle execution differs')
    return result


def audit(source):
    execution=asyncio.run(numerical(source))
    baseline=json.loads((ROOT/'docs/reviews/physics_remaining_source_pairs.json').read_text())
    pairs=[g for g in baseline['graphs'] if g['derivation_ids']==['000016']]
    require(len(pairs)==1,'One sine_double_angle source pair required');pair=pairs[0]
    versions=[pair['legacy_version_id'],pair['projected_version_id']]
    require(versions[1]==SOURCE_VERSION,'Qualified projected source differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph,parent=check_parent(db);inventories=[];forms=[];sources=[]
        for version,key in zip(versions,['legacy','projected']):
            row=db.execute('SELECT a.artifact_id::text,v.version_id::text,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(version,)).fetchone()
            require(row and row['artifact_id']==pair[key+'_artifact_id'] and row['content_hash']==pair[key+'_content_hash'],'Original sine_double_angle identity changed')
            nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(version,)).fetchall()
            edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id',(version,)).fetchall()
            bindings=db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',(version,)).fetchall()
            forms.append(scope(nodes,edges));inventories.append(source_inventory([json.loads(n['type_signature']) for n in nodes],bindings));sources.append(row)
        a,b=forms
        require(signature(a[0])==signature(b[0]) and a[2]==b[2] and {k:Counter(v) for k,v in a[1].items()}=={k:Counter(v) for k,v in b[1].items()},'Complete sine_double_angle grouped scope differs')
        require(inventories[0]['relations']==inventories[1]['relations'] and inventories[0]['binding_versions']==inventories[1]['binding_versions']
            and all(i['all_bindings_active'] for i in inventories),'Sine double angle inference/binding contracts differ')
    reports=[]
    for key,row in zip(['legacy','projected'],sources):
        reports.append(dict(family='sine_double_angle_'+key,legacy_artifact_id=row['artifact_id'],legacy_version_id=row['version_id'],legacy_content_hash=row['content_hash'],
            projected_source_version_id=SOURCE_VERSION,approved_execution_version_id=parent['version_id'],approved_execution_graph_sha256=parent['graph_sha256'],
            qualification=parent,source_steps=len(a[0]),complete_grouped_scope_sha256=digest(signature(a[0])),complete_pair_multiplicity=True,
            complete_dependency_edges=True,new_atoms_required=0))
    return dict(read_only=True,approved=False,catalog_mutations=0,graphs=reports,fresh_execution=execution,
        auditor_sha256=sha(__file__),parent_checker_sha256=sha(ROOT/'scripts/check_sine_double_angle_identity_parent.py'),
        limitations=['Exact imaginary-unit source identity is explicitly interpreted as I; historical source classifications remain preserved.',
            'Numerical realization accepts finite real radians converted to float64 and preserves shape; no degree or complex-input semantics.',
            'High-precision product-identity evaluation is compared with an independent direct double-angle oracle; no universal physical applicability claim.'])



if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-directory',type=Path,required=True)
    report=audit(p.parse_args().source_directory)
    (ROOT/'docs/reviews/sine_double_angle_identity_execution_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=False,graphs=len(report['graphs']),new_atoms=0)))
