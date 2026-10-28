"""Qualify original orbital-radius identities with their pinned rule discrepancy."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_remaining_legacy_physics_revisions import ROOT, sha, require, history
from scripts.check_orbit_radius_identity_parent import check_parent
from scripts.reconcile_orbit_radius_originals import numerical, public_step, STEP, PROJECTED_RULE
from scripts.audit_legacy_physics_execution_scope import scope
from scripts.audit_physics_projection_lineage import signature, digest
from sciona.physics_ingest.orbit_radius_proof import SOURCE_VERSION


def audit(source):
    execution = asyncio.run(numerical(source))
    path = ROOT/'docs/reviews/orbit_radius_original_reconciliation.json'
    retained = json.loads(path.read_text())
    require(retained['passed'] and retained['read_only'] and retained['catalog_mutations'] == 0
            and retained['reconciler_sha256'] == sha(ROOT/'scripts/reconcile_orbit_radius_originals.py')
            and retained['immutable_validator_sha256'] == sha(ROOT/'scripts/validate_orbit_radius_immutable_source.py'), 'Reconciliation changed')
    require(execution == retained['fresh_execution'] and public_step(source, execution) == retained['explicit_rule_correction'], 'Fresh source reconciliation differs')
    pairs = [g for g in json.loads((ROOT/'docs/reviews/physics_remaining_source_pairs.json').read_text())['graphs'] if g['derivation_ids'] == ['282755']]
    require(len(pairs) == 1 and pairs[0]['projected_version_id'] == SOURCE_VERSION, 'Exact source pair required')
    pair = pairs[0]; forms = []; reports = []
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph, parent = check_parent(db)
        for kind in ['legacy', 'projected']:
            prior = next(r for r in retained['originals'] if r['kind'] == kind)
            version = pair[kind+'_version_id']
            row = db.execute('SELECT artifact_id::text,version_id::text,content_hash FROM artifact_versions WHERE version_id=%s', (version,)).fetchone()
            require(row == {k: prior[k] for k in ['artifact_id','version_id','content_hash']}
                    and row == dict(artifact_id=pair[kind+'_artifact_id'], version_id=version, content_hash=pair[kind+'_content_hash'])
                    and history(db,version) == prior['original_history_sha256'], 'Reconciled original history changed')
            nodes = db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s', (version,)).fetchall()
            edges = db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s', (version,)).fetchall()
            form = scope(nodes,edges); forms.append(form)
            require(digest(signature(form[0])) == prior['grouped_scope_sha256'] and len(form[0]) == 13 and len(form[2]) == 12, 'Original grouped scope differs')
            reports.append(dict(family='orbit_radius_'+kind, legacy_artifact_id=row['artifact_id'], legacy_version_id=version,
                legacy_content_hash=row['content_hash'], projected_source_version_id=SOURCE_VERSION,
                approved_execution_version_id=parent['version_id'], approved_execution_graph_sha256=parent['graph_sha256'],
                qualification=parent, source_steps=13, complete_pair_multiplicity=True, complete_dependency_edges=True,
                exact_original_rule_parity=False, explicit_rule_correction=retained['explicit_rule_correction'],
                original_grouped_scope_sha256=prior['grouped_scope_sha256'], reconciled_scope_sha256=retained['reconciled_scope_sha256'],
                reconciliation_sha256=sha(path), new_atoms_required=0))
        a,b = forms; sa,sb = signature(a[0]),signature(b[0])
        require({k for k in sa.keys() | sb.keys() if sa.get(k) != sb.get(k)} == {str(STEP)}, 'Unexpected rule discrepancy')
        sa[str(STEP)]['rule'] = PROJECTED_RULE
        require(sa == sb and digest(sb) == retained['reconciled_scope_sha256'], 'Reconciled scope differs')
        require({k:Counter(v) for k,v in a[1].items()} == {k:Counter(v) for k,v in b[1].items()} and a[2] == b[2], 'Pair multiplicities or edges differ')
    return dict(read_only=True, approved=False, catalog_mutations=0, graphs=reports, fresh_execution=execution,
        reconciliation_sha256=sha(path), reconciler_sha256=sha(ROOT/'scripts/reconcile_orbit_radius_originals.py'),
        auditor_sha256=sha(__file__), parent_checker_sha256=sha(ROOT/'scripts/check_orbit_radius_identity_parent.py'),
        limitations=['Newtonian circular exterior test-mass model; positive finite G, central mass and period with identical nonempty shapes.',
            'Preserve the original four-variable rule with six feeds and the projected three-variable correction explicitly; no literal source parity.',
            'Correct source aliases and interpret pi as the mathematical constant; positive-domain cancellation and cube root require the stated assumptions.',
            'Radius is measured from the center, not surface altitude. Caller establishes clearance; geostationarity also needs equatorial prograde circular geometry and sidereal period.',
            'Reuse the unchanged provider for radius and orbital speed with independent output rounding; no finite two-body, perturbation, relativistic or station-keeping claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    report = audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/orbit_radius_identity_execution_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, graphs=2, new_atoms=0)))
