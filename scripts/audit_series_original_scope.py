"""Fresh source replay and conditional numerical review for the series source graph."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
import sympy as sp

from sciona.atoms.electrical.series_resistance import series_resistance
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.pdg_graph_replay import replay_derivation
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='4e6f9e22-7b7c-5ae5-b9bf-20668cb743a2'
ORIGINAL='6e8e13f5-3652-5dc6-8be7-dfad57165a7a'
EXECUTION='3d6b613f-19b2-5b5e-92f2-ffcdfdf63916'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(source):
    symbols=(source/'symbols.cypher').read_bytes();rules_bytes=(source/'infrules.cypher').read_bytes()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='pdg-graph-replay.v1' AND passed",(ORIGINAL,)).fetchall()
        require(len(rows)==1,'One successful source replay required');recorded=rows[0]['details']
        original=db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s',(ORIGINAL,)).fetchone()
        require(str(original['artifact_id'])==ARTIFACT and original['content_hash']==recorded['graph_content_hash'],'Original identity differs')
        nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(ORIGINAL,)).fetchall()
        edges=db.execute('SELECT * FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id',(ORIGINAL,)).fetchall()
        projection={key:[{k:str(v) if k=='version_id' else v for k,v in row.items()} for row in values] for key,values in [('nodes',nodes),('edges',edges)]}
        require(_digest(projection)==recorded['graph_projection_sha256'],'Original proof projection changed')
        for name,digest in recorded['implementation_hashes'].items():require(sha(ROOT/name)==digest,'Replay implementation changed')
        resolved={};dimensions={};expression_hashes={}
        for identity,digest in recorded['expression_evidence_sha256'].items():
            row=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE expression_id=%s',(identity,)).fetchone()
            fresh=prepare_pdg_evidence(row,symbols)
            require(fresh==row['evidence_json']['pdg_source_comparison'] and _digest(fresh)==digest,'Source expression changed')
            require(fresh['source_comparison']['correspondence']=='exact_ast_match','Exact source correspondence required')
            resolved[identity]=deserialize_expr(fresh['upstream_symbolic']['sympy_srepr']);expression_hashes[identity]=digest
            for variable in fresh['source_variable_dimensions']:
                name=variable['source_symbol'];dimension=variable['dim_signature']
                require(name not in dimensions or dimensions[name]==dimension,'Ambiguous source dimensions')
                dimensions[name]=dimension
        replay=replay_derivation(nodes,resolved,load_pinned_rule_contracts(rules_bytes,recorded['rule_file_sha256']),
            edges=[(e['source_id'],e['target_id']) for e in edges])
        for key in replay:require(replay[key]==recorded[key],'Fresh replay differs: '+key)
        require(len(replay['steps'])==4 and len(replay['terminal_expression_ids'])==1,'Complete four-step source required')
        terminal=deserialize_expr(replay['steps'][-1]['computed_srepr'])
        current=sp.Symbol('pdg0004501');a=sp.Symbol('pdg0003461');b=sp.Symbol('pdg0008697')
        require(sp.cancel(terminal.rhs-a-b)==0 and terminal.lhs==sp.Symbol('pdg0001908'),'Terminal scope differs')
        require(replay['required_conditions']==[sp.srepr(sp.Ne(current,0,evaluate=False))],'Nonzero-current proof condition differs')
        resistance=DimensionalSignature(M=1,L=2,T=-3,I=-2).to_compact()
        require(all(dimensions.get(str(symbol))==resistance for symbol in [a,b,terminal.lhs])
            and dimensions.get(str(current))==DimensionalSignature(I=1).to_compact(),'Source resistance/current dimensions differ')
        evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='series-cdg-execution.v1' AND passed",(ORIGINAL,)).fetchall()
        require(len(evidence)==1 and sha(inspect.getfile(series_resistance))==evidence[0]['details']['provider_implementation_sha256'],'Qualified provider changed')
        parent=db.execute('SELECT a.status,a.is_publishable,v.is_latest,v.trust_tier,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Approved derivative required')
    rng=np.random.default_rng(20260910)
    first=10**rng.uniform(-3,4,256);second=10**rng.uniform(-3,4,256)
    currents=rng.uniform(.01,2,256)*rng.choice([-1,1],256)
    matrix=np.array([[1.,0.,0.],[0.,1.,0.],[-1.,-1.,1.]])
    voltages=np.linalg.solve(matrix,np.stack([currents*first,currents*second,np.zeros(256)]))
    reference=voltages[2]/currents;actual=series_resistance(first,second,currents)
    np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(series_resistance(second,first,currents),actual)
    np.testing.assert_array_equal(series_resistance(first,second,-2*currents),actual)
    for values in [(first,second,np.zeros(256)),(-first,second,currents)]:
        try:series_resistance(*values)
        except ValueError:pass
        else:raise ValueError('Required domain rejection missing')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,
        original_graph_sha256=recorded['graph_content_hash'],approved_execution_version_id=EXECUTION,
        approved_execution_graph_sha256=parent['content_hash'],fresh_replay=replay,expression_evidence_sha256=expression_hashes,
        source_dimensions_verified=True,synthetic_cases=256,maximum_relative_error=float(np.max(np.abs(actual-reference)/np.abs(reference))),
        numerical_checks=['Independent Kirchhoff linear-system reference','Resistor permutation','Current scaling and reversal','Zero-current rejection','Negative-resistance rejection'],
        provider_source_sha256=sha(inspect.getfile(series_resistance)),
        scope='Two lumped ohmic resistors in one series branch, with common nonzero current and additive voltage drops. The numerical terminal relation is R_eq=R_a+R_b.',
        tier_policy='Eligible evidence for conditional automated Tier 3 review. Historical needs_human bounds are preserved; they are not a requirement for lower-tier certification. Any new execution version must carry its own automated regime review.',
        reuse='Reuse the existing provider wherever this two-component ohmic model and SI units apply; no source-specific records or hardware identifiers are required. Other additive domains need their own dimensioned adapters.',
        limitations=['Physical topology and ohmic behavior are caller-supplied model assumptions, not inferred from numeric arrays.',
            'This audit executes the approved provider, not a newly staged original-identity graph.',
            'No Tier 1 human certification, nonlinear-device claim or zero-current derivation extension.'],
        source_sha256={'symbols.cypher':hashlib.sha256(symbols).hexdigest(),'infrules.cypher':hashlib.sha256(rules_bytes).hexdigest()},
        validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_series_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_expressions=len(report['expression_evidence_sha256']),proof_steps=4,synthetic_cases=256,catalog_mutations=0)))
