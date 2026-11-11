"""Offline fault injection for the complete-stack review's pre-catalog gates.

The database boundary is a sentinel, never a simulated successful catalog.
Passing this check does not qualify lineage, staging or publication.
"""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts import review_stacking_complete_scope as reviewer


class CatalogBoundaryReached(Exception):
    pass


def main():
    original = Path.read_text
    completed = []

    def baseline():
        with patch.object(reviewer.psycopg, 'connect', side_effect=CatalogBoundaryReached) as connection:
            try:
                reviewer.review()
            except CatalogBoundaryReached:
                if connection.call_count != 1:
                    raise ValueError('Expected exactly one catalog boundary')
                return
        raise ValueError('Reviewer unexpectedly bypassed the catalog boundary')

    def reject(name, filename, mutate, expected):
        target = reviewer.ROOT / 'docs/reviews' / filename

        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path.resolve() == target.resolve():
                document = json.loads(text)
                mutate(document)
                return json.dumps(document)
            return text

        with patch.object(Path, 'read_text', changed), patch.object(
                reviewer.psycopg, 'connect', side_effect=CatalogBoundaryReached) as connection:
            try:
                reviewer.review()
            except ValueError as error:
                if expected not in str(error) or connection.called:
                    raise ValueError('Unexpected rejection for ' + name) from error
                completed.append(name)
                return
        raise ValueError('Weakened evidence reached the catalog boundary: ' + name)

    baseline()
    execution = 'competition_otto_decomposed_complete_execution.json'
    semantic = 'competition_otto_semantic_review.json'
    for name, key, value, expected in (
        ('failed_native_execution', 'status', 'failed', 'Unapproved complete execution'),
        ('premature_approval', 'approved', True, 'Unapproved complete execution'),
        ('unexpected_catalog_writes', 'catalog_mutations', 1, 'Unapproved complete execution'),
        ('different_graph', 'serialized_graph_sha256', '0'*64, 'Complete graph identity'),
        ('missing_nodes', 'actual_runner_nodes', 61, 'Complete native qualification'),
        ('partial_first_level', 'source_entries', 32, 'Complete native qualification'),
        ('partial_supplemental', 'supplemental_blocks', 6, 'Complete native qualification'),
        ('reduced_tuning', 'native_meta_tuning_fits', 8799, 'Complete native qualification'),
        ('reduced_refits', 'native_meta_refits', 1099, 'Complete native qualification'),
        ('failed_source_parity', 'retained_blend_exact_match', False, 'Complete native qualification'),
        ('weakened_precision', 'decimal_oracle_precision', 16, 'Complete native qualification'),
        ('weakened_tolerance', 'decimal_oracle_rtol', 1e-3, 'Independent numerical tolerance'),
    ):
        reject(name, execution, lambda d, k=key, v=value: d.__setitem__(k, v), expected)
    reject('missing_native_bag', execution,
           lambda d: d['completed_native_meta_bags'].pop(), 'Complete meta execution inventory')
    reject('reduced_native_bag', execution,
           lambda d: d['completed_native_meta_bags'][0].__setitem__('fitted_models', 1),
           'Full native family bags')
    reject('different_fold_population', execution,
           lambda d: d['completed_native_meta_bags'][0].__setitem__('fitting_rows', 1011),
           'Native fold/refit populations')
    reject('missing_implementation', execution,
           lambda d: d['implementation_sha256'].pop('sciona/stacking_complete_graph.py'),
           'Incomplete implementation coverage')
    reject('implementation_drift', execution,
           lambda d: d['implementation_sha256'].__setitem__('sciona/stacking_complete_graph.py', '0'*64),
           'Evidence drift')
    reject('runtime_evidence_drift', execution,
           lambda d: d.__setitem__('candidate_runtime_evidence_sha256', {}), 'Native candidate closure')
    reject('provider_drift', execution,
           lambda d: d['provider_sha256'].__setitem__('sciona.atoms.ml.stacking_complete', '0'*64),
           'Provider code changed')
    reject('different_original_source', semantic,
           lambda d: d.__setitem__('source_hash', '0'*64), 'Retained source identity')
    reject('missing_source_stage', semantic,
           lambda d: d['intake_stage_mapping'].pop('stratification'), 'Retained six-stage corrections')
    baseline()
    paths = ['scripts/review_stacking_complete_scope.py',
             'scripts/validate_stacking_complete_review_gates.py',
             'docs/reviews/competition_otto_decomposed_complete_execution.json']
    report = dict(status='offline_review_gates_passed', approved=False, catalog_mutations=0,
                  real_database_connections=0, pre_catalog_baseline_passed_before_and_after=True,
                  rejected_scenarios=completed,
                  sha256={p: hashlib.sha256((reviewer.ROOT/p).read_bytes()).hexdigest() for p in paths},
                  limitations=['Database access replaced by an explicit stop sentinel.',
                               'No catalog lineage, stored-graph, rollback or publication qualification.',
                               'Faults change in-memory review inputs; retained evidence bytes stay unchanged.'])
    destination = reviewer.ROOT/'docs/reviews/competition_otto_decomposed_review_gates.json'
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], rejected_scenarios=len(completed))))


if __name__ == '__main__':
    main()
