"""Review exact identities and interfaces for a non-publishable numerical draft."""
import hashlib
import importlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_residual_classifier_bindings import review

ROOT = Path(__file__).resolve().parents[1]
FQDN = 'cdg.ml.calibration.grouped_residual_classifier'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    current = review()
    for binding in current['bindings']:
        binding.pop('current_catalog')
    directory = ROOT/'docs/reviews'
    execution = json.loads((directory/'residual_classifier_bound_execution.json').read_text())
    reuse = json.loads((directory/'residual_classifier_reuse.json').read_text())
    if not execution['passed'] or not reuse['passed'] or execution['graph_sha256'] != reuse['graph_sha256'] or reuse['graph_sha256'] != current['graph_sha256']:
        raise ValueError('Matching runtime and reuse evidence required')
    if execution['binding_review_sha256'] != sha(directory/'residual_classifier_binding_review.json'):
        raise ValueError('Binding evidence drift')
    if execution['dependency_review_sha256'] != sha(directory/'residual_classifier_provider_dependencies.json'):
        raise ValueError('Dependency evidence drift')
    for name, expected in execution['execution_source_sha256'].items():
        path = Path(importlib.import_module(name).__file__).resolve() if name.startswith('sciona.atoms.ml.') else ROOT/name
        if sha(path) != expected:
            raise ValueError('Execution source drift: '+name)
    for path, expected in [('scripts/qualify_residual_bound_execution.py', execution['qualifier_sha256']),
                           ('scripts/validate_residual_classifier_reuse.py', reuse['validator_sha256'])]:
        if sha(ROOT/path) != expected:
            raise ValueError('Validator drift')
    current.update(scope='Complete grouped two-stage regression, residual classifier calibration and corrected integer prediction on supplied numeric feature matrices. Draft only; license and publication gates remain open.',
        evidence_sha256={name:sha(directory/name) for name in [
            'residual_classifier_bound_execution.json', 'residual_classifier_reuse.json',
            'residual_classifier_graph_witnesses.json', 'residual_classifier_dependency_notices.json']},
        planner_sha256=sha(Path(__file__)))
    return current


def merge_ports(nodes, direction):
    """Preserve each tested dtype and contextual contract, with ordinal aliases."""
    selected = [getattr(node, direction) for node in nodes]
    if len({len(ports) for ports in selected}) != 1:
        raise ValueError('Conflicting callable arity')
    merged = []
    for ordinal in range(len(selected[0])):
        variants = [ports[ordinal].model_dump() for ports in selected]
        first = dict(variants[0])
        if direction == 'inputs' and len({p['name'] for p in variants}) != 1:
            raise ValueError('Conflicting callable parameter names')
        for key in ['required', 'default_value_repr', 'dim_signature']:
            if any(p[key] != first[key] for p in variants):
                raise ValueError('Conflicting interface field: '+key)
        first['type_desc'] = ' | '.join(dict.fromkeys(p['type_desc'] for p in variants))
        first['constraints'] = 'Tested graph contexts: ' + ' '.join(dict.fromkeys(p['constraints'] for p in variants))
        merged.append(first)
    return merged


def plan():
    semantic = audit()
    graph = build_residual_classifier_graph()
    digest = encode_execution_graph(graph)[0]
    identity = uuid5(NAMESPACE_URL, 'sciona-reusable-cdg:'+FQDN)
    atoms, bindings = [], []
    for reviewed in semantic['bindings']:
        nodes = [node for node in graph.nodes if node.matched_primitive == reviewed['runtime']]
        atom_id = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+reviewed['fqdn'])
        atom = dict(fqdn=reviewed['fqdn'], artifact_id=str(atom_id), version_id=reviewed['version_id'],
                    content_hash=reviewed['content_hash'], runtime_fqdn=reviewed['runtime'],
                    inputs=merge_ports(nodes, 'inputs'), outputs=merge_ports(nodes, 'outputs'))
        atoms.append(atom)
        for node in nodes:
            bindings.append(dict(atom, node_id=node.node_id,
                                 inputs=[p.model_dump() for p in node.inputs],
                                 outputs=[p.model_dump() for p in node.outputs]))
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for item in atoms+[dict(fqdn=FQDN, artifact_id=str(identity))]:
            for table, key in [('artifacts', 'artifact_id')]+([('atoms', 'atom_id')] if item['fqdn'] != FQDN else []):
                # Table and column names above are fixed literals, never user input.
                rows = db.execute(f'SELECT {key} AS identity FROM {table} WHERE fqdn=%s', (item['fqdn'],)).fetchall()
                if len(rows)>1 or any(str(row['identity']) != item['artifact_id'] for row in rows):
                    raise ValueError('Existing catalog identity conflict: '+item['fqdn'])
    connected = {(edge.target_id, edge.input_name) for edge in graph.edges}
    boundary = {}
    for node in graph.nodes:
        for port in node.inputs:
            if (node.node_id, port.name) not in connected:
                value = port.model_dump()
                if port.name in boundary and boundary[port.name] != value:
                    raise ValueError('Ambiguous external input')
                boundary[port.name] = value
    return dict(fqdn=FQDN, artifact_id=str(identity), version_id=str(uuid5(identity, digest)),
        graph_sha256=digest, atoms=atoms, bindings=bindings,
        boundary_inputs=list(boundary.values()),
        boundary_outputs=[p.model_dump() for p in next(n for n in graph.nodes if n.node_id=='final').outputs],
        mandatory_provenance=dict(fqdn='cdg.competition.solution.kaggle.nasa_airport_pushback_phase_1_3rd_place_1st',
            scope='Mandatory original intake provenance; numerical reconstruction only, not completion or approval of the domain workflow.'))


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/residual_classifier_draft_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(atoms=len(result['atoms']), bindings=len(result['bindings']),
        boundary_inputs=len(result['boundary_inputs']), approved=False, catalog_mutations=0)))
