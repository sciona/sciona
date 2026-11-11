"""Execute the catalog-stored candidate against the unchanged pinned-source oracle."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from scripts import validate_causal_full_decomposition as oracle
from scripts.stage_causal_corrected_intake import (
    ROOT, PLAN, qualified_plan, connect, check_candidate, sha,
)


async def validate(reference_dir):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Explicit single-thread environment required')
    plan, providers = qualified_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph = check_candidate(db, plan, providers)
    retained_path = ROOT / 'docs/reviews/causal_full_decomposition_source_parity.json'
    retained = json.loads(retained_path.read_text())
    for path, digest in retained['implementation_hashes'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Qualified implementation changed: ' + path)
    previous = oracle.build_explicit_causal_feature_graph
    oracle.build_explicit_causal_feature_graph = lambda: graph.model_copy(deep=True)
    try:
        result = await oracle.validate(reference_dir)
    finally:
        oracle.build_explicit_causal_feature_graph = previous
    for field in ('source_commit', 'source_hashes', 'implementation_hashes',
                  'provider_hashes', 'library_versions', 'cases', 'nodes', 'edges'):
        if result[field] != retained[field]:
            raise ValueError('Stored execution differs from qualification: ' + field)
    if result['graph_hash'] != plan['graph_sha256']:
        raise ValueError('Executed candidate hash differs')
    after, after_providers = qualified_plan()
    if after != plan:
        raise ValueError('Qualification changed during execution')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if check_candidate(db, after, after_providers) != graph:
            raise ValueError('Stored candidate changed during execution')
    result.update(approved=False, catalog_mutations=0, stored_candidate=True,
                  candidate_version_id=plan['version_id'], original_artifact_id=plan['artifact_id'],
                  plan_sha256=sha(PLAN), validator_sha256=sha(__file__),
                  candidate_checker_sha256=sha(ROOT / 'scripts/stage_causal_corrected_intake.py'),
                  retained_base_evidence_sha256=sha(retained_path),
                  limitations=[
                      'Synthetic same-library parity; no historical-library, predictive-quality or causal-validity claim.',
                      'Original CDG remains unpublished pending publication gates.',
                      'Generic numerical providers retain their reviewed bounds; feature ordering and class roles are recipe adapters.',
                  ])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(args.reference_dir))
    (ROOT / 'docs/reviews/causal_corrected_intake_execution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'nodes', 'cases', 'candidate_version_id')}), flush=True)
