"""Reconcile the original radius derivation under explicit positive real assumptions."""
import argparse
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
import sympy as sp

from scripts.validate_physics_real_power_replay import validate
from sciona.ghost.symbolic import deserialize_expr

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='65b11a12-8737-5a29-b63c-924658b4dc9c'
ORIGINAL='200fb45c-6efe-5fed-aadc-7aff67423eb6'
REPLAY='f05dd820-e273-506d-87a3-625d4cd08e76'
EXECUTION='3432207b-0944-5f6d-9fcd-4778c3f45346'


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(source):
    refreshed=validate(ROOT,source/'symbols.cypher',source/'infrules.cypher')
    proofs=[r for r in refreshed['graphs'] if r['version_id']==REPLAY]
    require(len(proofs)==1 and proofs[0]['passed'],'Fresh source proof required')
    proof=proofs[0];comparisons=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        original=db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s',(ORIGINAL,)).fetchone()
        require(original and str(original['artifact_id'])==ARTIFACT and original['content_hash']=='caac64334cd919f0a6b552445b59e6df53a346d955fc46c9a5396257dd237e8a','Original identity differs')
        by_version={}
        for version in [ORIGINAL,REPLAY]:
            rows=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(version,)).fetchall()
            by_version[version]={r['node_id']:json.loads(r['type_signature']) for r in rows}
        require(set(by_version[ORIGINAL])==set(by_version[REPLAY]) and len(by_version[ORIGINAL])==4,'Complete original four-step scope required')
        mapping=next(iter(by_version[REPLAY].values()))['source_expression_substitutions']
        for key,old in by_version[ORIGINAL].items():
            new=by_version[REPLAY][key]
            require(new['source_expression_substitutions']==mapping and new['inference_rule_id']==old['inference_rule_id']
                and new['variable_bindings']==old['variable_bindings'],'Inference rules or feeds differ')
            for field in ['inputs','outputs']:
                require([mapping[i] for i in old[field]]==new[field],'Expression topology differs')
        require(set(mapping.values())==set(proof['expression_evidence_sha256']),'Complete expression reconciliation required')
        for old,new in sorted(mapping.items()):
            records={}
            for identity in [old,new]:
                records[identity]=db.execute('SELECT sympy_srepr,evidence_json FROM artifact_symbolic_expressions WHERE expression_id=%s',(identity,)).fetchone()
            dimensions=records[new]['evidence_json']['pdg_source_comparison']['source_variable_dimensions']
            replacements={sp.Symbol(v['symbol_name']):sp.Symbol(v['source_symbol'],positive=True) for v in dimensions}
            first=deserialize_expr(records[old]['sympy_srepr']);second=deserialize_expr(records[new]['sympy_srepr'])
            require(first.free_symbols==set(replacements) and second.free_symbols==set(replacements),'Complete scalar identity mapping required')
            require(isinstance(first,sp.Equality) and isinstance(second,sp.Equality),'Source equations required')
            require(sp.simplify((first.lhs-second.lhs).xreplace(replacements))==0
                and sp.simplify((first.rhs-second.rhs).xreplace(replacements))==0,'Positive-domain source equivalence failed')
            comparisons.append(dict(original_expression_id=old,source_expression_id=new,positive_real_equivalence=True,
                original_expression_sha256=hashlib.sha256(records[old]['sympy_srepr'].encode()).hexdigest(),
                source_evidence_sha256=proof['expression_evidence_sha256'][new]))
        parent=db.execute('SELECT a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Approved numerical derivative required')
    for serialized in proof['required_conditions']:
        condition=deserialize_expr(serialized)
        require(sp.simplify(condition.xreplace({s:sp.Symbol(str(s),positive=True) for s in condition.free_symbols}))==sp.true,'Positive regime does not imply proof condition')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,
        source_replay_version_id=REPLAY,approved_execution_version_id=EXECUTION,approved_execution_graph_sha256=parent['content_hash'],
        original_source_equivalence=comparisons,proof_steps=len(proof['steps']),positive_regime_conditions_verified=len(proof['required_conditions']),
        source_proof=proof,source_implementation_sha256=refreshed['implementation_sha256'],
        scope='Evaluate the terminal length scale 2GM/c^2 for positive SI mass, G and c. Original parsed expressions and normalized source expressions agree under positive real quantity assumptions.',
        reuse='The existing dimensioned provider computes the gravitational length scale from caller-supplied mass and constants; no dataset-specific input or fitted model is required.',
        limitations=['This verifies forward conditional source algebra, not Newtonian light-motion premises or general-relativistic field equations.',
            'Horizon interpretation requires caller-established spherical nonrotating uncharged Schwarzschild regime with an asymptotically flat vacuum exterior. No black-hole classification.',
            'Positivity reconciles the principal square-root representation without asserting reverse square-root inference. Historical failed evidence remains preserved.',
            'The source equation labels are reconciled to source-defined scalar identities; this is agreement of stored parsed equations, not an independent new LaTeX parser validation.'],
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_schwarzschild_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,expressions=len(report['original_source_equivalence']),proof_steps=report['proof_steps'],conditions=report['positive_regime_conditions_verified'],catalog_mutations=0)))
