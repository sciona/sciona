"""Read-only source reconciliation and fresh hyperbolic-identities execution; no promotion."""
import argparse
import asyncio
from collections import Counter
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.audit_physics_projection_lineage import grouped
from scripts.audit_physics_projection_lineage import signature, digest
from scripts.audit_physics_legacy_reconciliation import source_inventory
from scripts.stage_remaining_legacy_physics_revisions import ROOT, sha, require, history
from scripts.validate_hyperbolic_identities_immutable_source import validate as validate_source
from scripts import validate_hyperbolic_identities_execution as oracle
from scripts.verify_hyperbolic_identities_catalog import verify


MISSING_EDGES = {(('713234','3636266'),('713234','3919490')),
                 (('713234','6389859'),('713234','9722132'))}

def scope(nodes, edges):
    groups=grouped(nodes)
    owner={node:key for key,g in groups.items() for node in g['nodes']}
    require(len(owner)==len(nodes), 'Duplicate source node')
    require(all(e['source_id'] in owner and e['target_id'] in owner for e in edges), 'Unknown edge endpoint')
    actual={(owner[e['source_id']],owner[e['target_id']]) for e in edges}
    expected={(a,b) for a,x in groups.items() for b,y in groups.items() if x['outputs'] & y['inputs']}
    require(expected-actual==MISSING_EDGES and not actual-expected, 'Unreviewed source edge difference')
    require(len(groups)==18 and len(actual)==17, 'Source graph cardinality differs')
    pairs={key:[] for key in groups}
    for node in nodes:
        sig=json.loads(node['type_signature']);v=sig['variable_bindings'];key=(v['derivation_id'],v['step_id'])
        pairs[key].extend(itertools.product(sig['inputs'],sig.get('outputs',[sig['output']])))
    reviewed = [
        ('3636266', '3919490', '111182', '111253', 'ff9be36d-020b-5632-b718-b17a881a0955'),
        ('6389859', '9722132', '111355', '111253', '1c9f3d3c-96af-53c2-b749-a6518f55fb41'),
    ]
    for producer_id, consumer_id, producer_rule, consumer_rule, expression in reviewed:
        producer=groups[('713234',producer_id)];consumer=groups[('713234',consumer_id)]
        require(producer['rule']==producer_rule and consumer['rule']==consumer_rule
                and producer['outputs'] & consumer['inputs']=={expression},
                'Reviewed definition-to-self-product overlap differs')
    return groups,pairs,sorted(actual)


async def numerical(source):
    with patch.object(oracle, 'validate_source', side_effect=validate_source):
        result = await oracle.validate(ROOT, source/'symbols.cypher', source/'infrules.cypher', source/'expr_and_feed.cypher')
    normalized = json.loads(json.dumps(result))
    normalized['source_proof'].pop('immutable_validator_sha256')
    require(normalized == json.loads((ROOT/'docs/reviews/hyperbolic_identities_execution.json').read_text()),
            'Fresh hyperbolic-identities execution differs from retained evidence')
    return result


def reconcile(source):
    execution = asyncio.run(numerical(source))
    parent = verify(ROOT)
    pairs = [g for g in json.loads((ROOT/'docs/reviews/physics_remaining_source_pairs.json').read_text())['graphs']
             if g['derivation_ids'] == ['713234']]
    require(len(pairs) == 1, 'Unique hyperbolic-identities source pair required')
    pair = pairs[0]
    forms, inventories, originals = [], [], []
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for key in ('legacy', 'projected'):
            version = pair[key+'_version_id']
            row = db.execute('SELECT artifact_id::text,version_id::text,content_hash FROM artifact_versions WHERE version_id=%s', (version,)).fetchone()
            require(row == dict(artifact_id=pair[key+'_artifact_id'], version_id=version, content_hash=pair[key+'_content_hash']),
                    'Original hyperbolic-identities identity changed')
            nodes = db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s', (version,)).fetchall()
            edges = db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s', (version,)).fetchall()
            bindings = db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s', (version,)).fetchall()
            forms.append(scope(nodes, edges))
            inventories.append(source_inventory([json.loads(n['type_signature']) for n in nodes], bindings))
            originals.append(dict(kind=key, **row, nodes=len(nodes), steps=len(forms[-1][0]), bindings=len(bindings),
                                  original_history_sha256=history(db, version)))
    a, b = forms
    require(signature(a[0]) == signature(b[0]), 'Grouped step contracts differ')
    require({k: Counter(v) for k,v in a[1].items()} == {k: Counter(v) for k,v in b[1].items()}, 'Pair multiplicities differ')
    require(a[2] == b[2], 'Dependency edges differ')
    x, y = inventories
    require(x['relations'] == y['relations'] and x['binding_versions'] == y['binding_versions']
            and x['all_bindings_active'] and y['all_bindings_active'], 'Inference or binding contracts differ')
    return dict(passed=True, read_only=True, approved=False, eligible_for_approval=False, catalog_mutations=0,
                originals=originals, shared_source_steps=len(a[0]), complete_pair_multiplicity=True,
                complete_dependency_edges=False, stored_dependency_edges_equal=True,
                missing_expression_overlap_edges=sorted(MISSING_EDGES),
                source_edge_reconciliation="Both histories retain the same 17 edges and omit the stored step-eight to step-one and step-ten to step-two expression overlaps. Stored steps seven, eight and ten have inference directions that differ from the corrected exponential-definition proof. The retained reconstruction executes step eight before step seven and independently checks every reconstructed identity; neither historical graph is certified as a complete executable source proof.",
                shared_scope_sha256=digest(signature(a[0])),
                fresh_execution=execution, existing_parent_verification=parent,
                limitations=['Source correspondence is not original-identity execution approval.',
                    'The two exact expression-overlap discrepancies and corrected inference directions remain explicit; historical graph edges are not rewritten.',
                    'Finite real dimensionless argument; retain the full hyperbolic function and identity collection with exponential-definition certificates. Complex sine/cosine definitions are checked separately; no global complex quotient domain or floating-point cancellation stability claim.',
                    'Before promotion: exact parent/provider bounds and ports, corrected original versions, stored execution, corruption rejection, rollback and served execution.'],
                reconciler_sha256=sha(__file__),
                immutable_validator_sha256=sha(ROOT/'scripts/validate_hyperbolic_identities_immutable_source.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    report = reconcile(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/hyperbolic_identities_original_reconciliation.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps(dict(passed=True, approved=False, shared_steps=report['shared_source_steps'], catalog_mutations=0)))
