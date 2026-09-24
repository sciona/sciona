"""Version-bound automated Tier 3 review of staged lifecycle providers."""
import json
from pathlib import Path
from uuid import NAMESPACE_URL,uuid5

from scripts.plan_nasa_first_lifecycle_providers import ROOT,sha,evidence
from scripts.review_lifecycle_runtime_dependencies import review as dependency_review
from scripts.review_available_feature_publication import require


def plan():
    """Use the immutable staged cohort, including after it becomes served."""
    proposed=json.loads((ROOT/'docs/reviews/competition_nasa_first_lifecycle_provider_plan.json').read_text())
    require(proposed['evidence']==evidence(),'Staged qualification changed')
    contracts=json.loads((ROOT/'docs/reviews/competition_nasa_first_lifecycle_contracts.json').read_text())
    for name,digest in contracts['source_sha256'].items():require(sha(ROOT/name)==digest,'Contract code changed')
    expected=[dict(p,artifact_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+p['fqdn']))) for p in contracts['providers']]
    require(proposed['atoms']==expected and len(expected)==35,'Frozen provider cohort changed')
    for atom in expected:
        require(sha(ROOT.parent/'sciona-atoms-ml'/atom['provider_source'])==atom['provider_sha256'],'Provider implementation changed')
    return proposed


def review():
    proposed=plan();directory=ROOT/'docs/reviews'
    names=['competition_nasa_first_lifecycle_runtime.json','competition_nasa_first_lifecycle_stored_reuse.json',
        'competition_nasa_first_lifecycle_stored_adapters.json','competition_nasa_first_lifecycle_database_gates.json',
        'competition_nasa_first_lifecycle_staging_gates.json','competition_nasa_first_native_state.json']
    runtime,reuse,adapters,gates,staging,native=[json.loads((directory/name).read_text()) for name in names]
    require(all(r['passed'] for r in [runtime,reuse,adapters,gates,staging,native]),'Qualification failed')
    for report in [reuse,adapters,staging,native]:
        for name,digest in report['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Qualification source changed: '+name)
    for name in ['competition_nasa_first_guarded_graph.json','competition_nasa_first_primary_graph.json',
                 'competition_nasa_first_raw_training_graph.json','competition_nasa_first_policy_state_graphs.json']:
        report=json.loads((directory/name).read_text())
        require(report['passed'],'Underlying graph qualification failed')
        for path,digest in report['implementation_sha256'].items():
            require(sha(ROOT/path)==digest,'Underlying graph implementation changed: '+path)
        names.append(name)
    require(gates['validator_sha256']==sha(ROOT/'scripts/validate_nasa_first_lifecycle_database_gates.py'),'Gate validator changed')
    require(gates['plan_sha256']==sha(directory/'competition_nasa_first_lifecycle_provider_plan.json'),'Gate plan changed')
    expected_faults=[]
    for i in range(35):
        for table in ['artifact_versions','atom_versions']:expected_faults.extend([f'{table}_hash_{i}',f'{table}_tier_{i}'])
        expected_faults.extend(f'{table}_contract_{i}' for table in ['artifact_io_specs','atom_io_specs'])
    expected_faults+=['missing_staging_evidence','evidence_payload','failed_audit','runtime_pointer','canonical_latest',
        'legacy_latest','canonical_publishable','legacy_publishable','source_hash']
    require(gates['rejected_faults']==expected_faults and gates['rollback_verified'],'Database fault coverage differs')
    require(staging['dry_run_rows']==895 and staging['failure_after_write_checks']==[1,447,895]
        and staging['original_and_target_rows_unchanged'],'Draft rollback coverage differs')
    expected={a['runtime_fqdn']:{k:a[k] for k in ['version_id','content_hash']} for a in proposed['atoms']}
    require(adapters['bindings']==expected and adapters['stored_domain_adapters']==19
        and adapters['executed_stored_providers']==35 and adapters['exact_prepared_rows']==245760
        and adapters['native_prediction_comparisons']==640,'Stored lifecycle execution coverage differs')
    generic={a['runtime_fqdn'].split('.model_selection.')[1]:{k:a[k] for k in ['version_id','content_hash']}
        for a in proposed['atoms'] if a['role']=='reusable_operation'}
    require({name:{k:b[k] for k in ['version_id','content_hash']} for name,b in reuse['bindings'].items()}==generic,
        'Generic version-bound reuse coverage differs')
    require(reuse['provider_graphs']==16 and reuse['executions']==38 and reuse['native_fits']==2
        and reuse['stored_ports_used'] and reuse['target_units']==['kWh','litre'],'Cross-domain execution differs')
    require(dependency_review()==runtime['dependencies'],'Provisioned runtime dependencies changed')
    require(runtime['excluded_optional_packages']==['httptools','psycopg_binary','uvloop','watchfiles']
        and runtime['psycopg_implementation']=='python','Runtime scope differs')
    return dict(format='nasa-first-lifecycle-publication-review.v1',review_source='automated',proposed_tier=3,
        eligible_for_approval_transaction=True,approved=False,catalog_mutations=0,
        provider_versions=[{k:a[k] for k in ['artifact_id','version_id','content_hash','runtime_fqdn','role']} for a in proposed['atoms']],
        qualification=proposed['evidence'],source_provenance=proposed['source_provenance'],
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(Path(__file__)),
        limitations=['Automated Tier 3 materialized computation only; no human Tier 1 or empirical Tier 2 claim.',
            'Sixteen generic operations permit cross-domain reuse with explicit target units, schema, time and caller policy contracts.',
            'Nineteen domain adapters retain source-specific conventions and must not be selected as generic domain-neutral transforms.',
            'Original competition workflow remains unapproved; publication here approves individual provider versions only.',
            'Native source-sized training has separate compositional evidence; stored training wiring uses a recording fit backend.',
            'Static symbolic propagation is unsupported. Select only for materialized execution.',
            'Provisioned in-process CPU profile only; clean installation, HTTP deployment and binary redistribution are unqualified.',
            'Model payloads and non-public input policies remain private runtime state; hashes establish integrity, not authentication.'])


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/competition_nasa_first_lifecycle_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,providers=len(report['provider_versions']),tier=3,approved=False)))
