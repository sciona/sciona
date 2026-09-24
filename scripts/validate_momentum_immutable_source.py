#!/usr/bin/env python3
"""Immutable source-version review for real-3D momentum reconstruction, independent of current artifact publication state."""
import argparse
import hashlib
import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
import sympy as sp

from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.momentum_vector_proof import SOURCE_VERSION, SOURCE_HASH, IDENTITIES, build_proof, vectors, verify_proof
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts, contract_blockers


def validate(root, symbol_file, rule_file):
    proof = build_proof()
    result = verify_proof(proof)
    symbol_bytes = symbol_file.read_bytes()
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.artifact_id::text,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if graph != dict(artifact_id='279f6f24-1db8-5ee3-a09f-f0283ca33ebb', content_hash=SOURCE_HASH):
            raise ValueError('Pinned original source identity differs')
        nodes = db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        edges = db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(nodes) != 5 or len(bindings) != 12 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Graph inventory differs')
        rows = {}
        for binding in bindings:
            selected = db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s', (binding['bound_artifact_fqdn'], binding['bound_version_content_hash'])).fetchall()
            if len(selected) != 1:
                raise ValueError('Unique source binding required')
            rows[str(selected[0]['expression_id'])] = selected[0]
        if len(rows) != 8:
            raise ValueError('Expected eight source equations')
        pins = {row['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'] for row in rows.values()}
        if len(pins) != 1:
            raise ValueError('Ambiguous inference pin')
        rules = load_pinned_rule_contracts(rule_file.read_bytes(), next(iter(pins)))
        source_records, interpreted = {}, {}
        v = vectors()
        for identity, row in rows.items():
            evidence = prepare_pdg_evidence(row, symbol_bytes)
            definitions = load_pinned_pdg_scalars(symbol_bytes, evidence['symbol_file_sha256'])
            for source in IDENTITIES:
                if definitions[source].dimension != DimensionalSignature(M=1, L=1, T=-1):
                    raise ValueError('Source momentum dimension differs')
                if not definitions[source].latex.startswith(r'\vec{p}'):
                    raise ValueError('Source vector declaration differs')
            raw = row['source_payload']['raw_payload']
            source_records[identity] = dict(version_id=str(row['version_id']), candidate_id=str(row['candidate_id']),
                candidate_payload_sha256=evidence['candidate_payload_sha256'], fresh_source_evidence_sha256=_digest(evidence),
                upstream_status=evidence['upstream_symbolic']['status'])
            if row['source_payload']['id'] == '6742123016':
                expected = dict(latex_lhs=r'\vec{p}_{electron}\cdot\vec{p}_{electron}',
                    latex_rhs=r'( \vec{p}_{1}\cdot\vec{p}_{1})+( \vec{p}_{2}\cdot\vec{p}_{2})-2( \vec{p}_{1}\cdot\vec{p}_{2})',
                    sympy_lhs="Symbol('pdg0004299')", sympy_rhs='', latex_relation='=')
                if any(raw.get(k) != value for k, value in expected.items()):
                    raise ValueError('Reviewed incomplete source expression differs')
                source_records[identity]['explicit_reconstruction'] = dict(source_latex=expected,
                    interpretation='Real 3D Euclidean dot products; exact reviewed LaTeX pattern, not scalar parsing.',
                    discrepancy='Source symbolic LHS is a vector identity instead of a squared norm; RHS is empty.')
                continue
            if evidence['upstream_symbolic']['status'] != 'roundtrip_passed':
                raise ValueError('Source linear vector AST unavailable')
            source = deserialize_expr(evidence['upstream_symbolic']['sympy_srepr'])
            if {str(s) for s in source.free_symbols}-set(IDENTITIES):
                raise ValueError('Unreviewed vector identity')
            interpreted[identity] = tuple(sp.Eq(source.lhs.xreplace({sp.Symbol(k): v[name][axis] for k, name in IDENTITIES.items()}),
                source.rhs.xreplace({sp.Symbol(k): v[name][axis] for k, name in IDENTITIES.items()}), evaluate=False) for axis in range(3))
        outputs, needed, operations = set(), set(), []
        for i, node in enumerate(nodes):
            if node['node_id'] != 'pdg_step_'+str(i+1):
                raise ValueError('Source step order differs')
            signature = json.loads(node['type_signature'])
            rule = rules.get(signature['inference_rule_id'])
            if not rule or contract_blockers(signature, rule):
                raise ValueError('Source rule arity differs')
            outputs.add(signature['output'])
            needed.update(signature['inputs'])
            if i < 4:
                equations = interpreted[signature['output']]
                if any(sp.expand(a-b) != 0 for actual, expected in zip(equations, proof.steps[i]) for a, b in zip(actual.args, expected.args)):
                    raise ValueError('Source vector conclusion differs')
            elif 'explicit_reconstruction' not in source_records[signature['output']]:
                raise ValueError('Expected missing dot-product source at final step')
            operations.append(dict(node_id=node['node_id'], rule=rule.name, inputs=signature['inputs'], output=signature['output']))
        roots = [interpreted[k] for k in needed-outputs]
        if len(roots) != 3 or any(sum(all(sp.expand(a-b) == 0 for actual, expected in zip(root, premise) for a, b in zip(actual.args, expected.args)) for root in roots) != 1 for premise in proof.premises):
            raise ValueError('Source vector premises differ')
        root_ids = []
        for premise in proof.premises:
            root_ids.append(next(k for k in needed-outputs if all(sp.expand(a-b) == 0
                for actual, expected in zip(interpreted[k], premise) for a, b in zip(actual.args, expected.args))))
        out = [operation['output'] for operation in operations]
        expected_inputs = [[root_ids[0], root_ids[1]], [out[0], root_ids[2]], [out[1]], [out[2]], [out[3], out[3]]]
        if [operation['inputs'] for operation in operations] != expected_inputs:
            raise ValueError('Source expression dependencies differ')
        expected_edges = {('pdg_step_'+str(i), 'pdg_step_'+str(i+1)) for i in range(1, 5)}
        if {(e['source_id'], e['target_id']) for e in edges} != expected_edges:
            raise ValueError('Source projected edges differ')
    files = ['scripts/validate_momentum_immutable_source.py', 'sciona/physics_ingest/momentum_vector_proof.py',
             'tests/physics_ingest/test_momentum_vector_proof.py', 'sciona/physics_ingest/pdg_evidence.py',
             'sciona/physics_ingest/pdg_symbols.py', 'sciona/physics_ingest/pdg_rule_contracts.py']
    result.update(read_only=True, approval_applied=False, selected_source_bindings=12, vector_equations_matched=7,
                  explicitly_reconstructed_dot_products=1, source_records=source_records, source_operations=operations,
                  symbol_file_sha256=hashlib.sha256(symbol_bytes).hexdigest(), rule_file_sha256=next(iter(pins)),
                  implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['checks', 'vector_equations_matched', 'explicitly_reconstructed_dot_products']}))
