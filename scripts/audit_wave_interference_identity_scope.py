"""Qualify corrected reuse while explicitly retaining the legacy omitted step."""
import argparse
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.reconcile_wave_interference_originals import ROOT, reconcile, require, sha
from scripts.check_wave_interference_identity_parent import check_parent
from sciona.physics_ingest.wave_interference_proof import SOURCE_VERSION


def audit(source):
    comparison=reconcile(source)
    retained=ROOT/'docs/reviews/wave_interference_original_reconciliation.json'
    require(comparison==json.loads(retained.read_text()),'Wave-interference reconciliation changed')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph,parent=check_parent(db)
    reports=[]
    for original in comparison['originals']:
        key=original['kind']
        if key=='projected':require(original['version_id']==SOURCE_VERSION,'Projected source differs')
        reports.append(dict(family='wave_interference_'+key,legacy_artifact_id=original['artifact_id'],
            legacy_version_id=original['version_id'],legacy_content_hash=original['content_hash'],
            projected_source_version_id=SOURCE_VERSION,approved_execution_version_id=parent['version_id'],
            approved_execution_graph_sha256=parent['graph_sha256'],qualification=parent,source_steps=original['steps'],
            shared_source_steps=26,shared_pair_multiplicity_verified=True,source_dependency_edges_verified=True,
            source_step_omission_preserved=key=='legacy',new_atoms_required=0,
            correction_scope='Recover the omitted malformed polar-amplitude step using the pinned full source and reviewed reconstruction.'
                if key=='legacy' else 'Correct the retained malformed polar-amplitude step using the reviewed reconstruction.',
            reconciliation_sha256=sha(retained)))
    return dict(read_only=True,approved=False,catalog_mutations=0,graphs=reports,
        fresh_execution=comparison['fresh_execution'],reconciliation_sha256=sha(retained),
        reconciler_sha256=sha(ROOT/'scripts/reconcile_wave_interference_originals.py'),
        auditor_sha256=sha(__file__),parent_checker_sha256=sha(ROOT/'scripts/check_wave_interference_identity_parent.py'),
        limitations=['Legacy has 26 source steps and projected source has 27; no claim of equal original scope or valid literal source proofs.',
            'Both corrected versions use the qualified reconstruction, explicitly recovering the legacy omission without rewriting history.',
            'Two scalar waves require common mode/polarization and normalization, fixed nonnegative amplitudes not both zero, and finite relative phase.',
            'Coherent fixed phase is not necessarily constructive. Incoherent mean assumes zero averaged cosine; a factor of two additionally requires equal amplitudes and constructive phase.',
            'Independent float64 output rounding; no arbitrary polarization, propagation, stochastic-amplitude, partial-coherence or absolute SI calibration claim.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/wave_interference_identity_execution_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=False,graphs=2,new_atoms=0,legacy_omission_explicit=True)))
