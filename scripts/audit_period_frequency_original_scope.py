"""Reconcile original period/frequency expressions with the pinned source proof."""
import argparse
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
import sympy as sp

from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest
from sciona.physics_ingest.pdg_graph_replay_v2 import replay_derivation
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='958b7895-6e01-55c0-8231-75c094b7e2c1'
ORIGINAL='e52a7e48-5681-5c6d-8626-4c156da09b48'
REPLAY='2280ed36-ecc7-52c3-b6e0-4e7b40931467'
EXECUTION='7c4e09cb-c51a-5fa0-be64-03b7ae73b204'


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(source):
    symbols=(source/'symbols.cypher').read_bytes();rules_bytes=(source/'infrules.cypher').read_bytes()
    T,f=sp.symbols('T f')
    original_ids=['0c5ff2f6-f581-504e-a3b7-895429529535','229c0272-2050-5a8c-bcdc-be8f5dbfb798','e91d63b7-aae2-5871-bb67-0980dd88fcfd']
    expected=[sp.Eq(T,1/f),sp.Eq(T*f,1),sp.Eq(f,1/T)]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        original=db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s',(ORIGINAL,)).fetchone()
        require(original and str(original['artifact_id'])==ARTIFACT and original['content_hash']=='4399bc7b4a035437d202f84df7fa58021410f062c9a7e00d1dd10d33f715cfae','Original version identity differs')
        original_nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(ORIGINAL,)).fetchall()
        require(len(original_nodes)==2,'Original two-step scope differs')
        for index,node in enumerate(original_nodes):
            signature=json.loads(node['type_signature'])
            require(signature['inputs']==[original_ids[index]] and signature['outputs']==[original_ids[index+1]]
                and signature['inference_rule_id']==['111182','111975'][index],'Original derivation topology differs')
        evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='pdg-source-graph-replay.v2' AND passed",(REPLAY,)).fetchall()
        require(len(evidence)==1,'Unique successful source replay required')
        recorded=evidence[0]['details'];resolved={};source_pins=set();comparisons=[]
        for filename,digest in recorded['implementation_hashes'].items():
            require(hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,'Replay implementation drift')
        for old,expected_equation in zip(original_ids,expected):
            new=recorded['expression_substitutions'][old]
            old_row=db.execute('SELECT sympy_srepr FROM artifact_symbolic_expressions WHERE expression_id=%s',(old,)).fetchone()
            old_expr=deserialize_expr(old_row['sympy_srepr'])
            require(isinstance(old_expr,sp.Equality) and sp.cancel(old_expr.lhs-expected_equation.lhs)==0
                and sp.cancel(old_expr.rhs-expected_equation.rhs)==0,'Original equation differs from reciprocal derivation')
            row=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifact_symbolic_expressions e '
                'JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE expression_id=%s',(new,)).fetchone()
            fresh=prepare_pdg_evidence(row,symbols)
            require(fresh==row['evidence_json']['pdg_source_comparison'] and _digest(fresh)==recorded['expression_evidence_sha256'][new],
                'Source expression evidence differs')
            require(fresh['source_comparison']['correspondence']=='exact_ast_match','Exact normalized source expression required')
            normalized=deserialize_expr(row['sympy_srepr'])
            require(isinstance(normalized,sp.Equality) and sp.cancel(normalized.lhs-expected_equation.lhs)==0
                and sp.cancel(normalized.rhs-expected_equation.rhs)==0,'Source equation differs from original computation')
            resolved[new]=deserialize_expr(fresh['upstream_symbolic']['sympy_srepr'])
            source_pins.add(row['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'])
            comparisons.append(dict(original_expression_id=old,source_expression_id=new,
                original_rational_equivalence=True,source_rational_equivalence=True,
                original_expression_sha256=hashlib.sha256(old_row['sympy_srepr'].encode()).hexdigest(),
                source_evidence_sha256=_digest(fresh)))
        require(len(source_pins)==1,'Unique source rule pin required')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(REPLAY,)).fetchall()
        edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s',(REPLAY,)).fetchall()
        replay=replay_derivation(nodes,resolved,load_pinned_rule_contracts(rules_bytes,next(iter(source_pins))),
            edges=[(e['source_id'],e['target_id']) for e in edges])
        for key in ['steps','required_conditions','root_expression_ids','terminal_expression_ids']:
            require(replay[key]==recorded[key],'Fresh proof differs: '+key)
        require(len(replay['steps'])==2 and len(replay['terminal_expression_ids'])==1,'Complete two-step proof required')
        allowed={sp.srepr(sp.Ne(sp.Symbol(name),0,evaluate=False)) for name in ['pdg0004201','pdg0009491']}
        require(set(replay['required_conditions'])==allowed,'Unexpected proof domain')
        selected=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a '
            'JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(EXECUTION,)).fetchone()
        require(selected and selected['status']=='approved' and selected['is_publishable'] and selected['is_latest']
            and selected['trust_tier']==3,'Approved numerical realization required')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(selected['fqdn'],)).fetchone()['d']
        graph=_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=selected['content_hash'],require_execution_envelope=True)
        require(len(graph.nodes)==1 and not graph.edges and graph.nodes[0].inputs[0].name=='period_seconds'
            and graph.nodes[0].outputs[0].name=='frequency_hz','Realization interface differs')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,
        source_replay_version_id=REPLAY,approved_execution_version_id=EXECUTION,execution_graph_sha256=selected['content_hash'],
        source_expression_comparisons=comparisons,fresh_replay=replay,
        original_scope='Derive ordinary frequency from cycle duration through T=1/f, T*f=1, f=1/T.',
        executable_scope='Evaluate the unique terminal frequency for supplied finite positive period in seconds; output finite positive hertz with preserved shape.',
        reconciliation='Original parsed and normalized source equations are rationally identical. Explicit multiplication by one explains the original strict AST mismatch; historical reports remain unchanged.',
        reuse='The same period-to-frequency implementation applies across periodic motion, oscillating circuits and recurring signals with explicit units. It does not estimate period from observations.',
        limitations=['Proof history remains the source of algebraic steps; the numerical execution graph evaluates the terminal relation.',
            'Positive finite period/frequency contracts imply both nonzero proof conditions. This is automated scoped Tier 3 evidence, not Tier 1 human review.',
            'This audit freshly replays algebra and checks the stored execution envelope; numerical measurements and candidate-version promotion require separate validation.'],
        source_sha256={'symbols.cypher':hashlib.sha256(symbols).hexdigest(),'infrules.cypher':hashlib.sha256(rules_bytes).hexdigest()},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    result=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_period_frequency_original_scope.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,expressions=len(result['source_expression_comparisons']),proof_steps=len(result['fresh_replay']['steps']),catalog_mutations=0)))
