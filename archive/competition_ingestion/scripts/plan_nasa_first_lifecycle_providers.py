"""Bind candidate lifecycle providers to current qualification for draft staging."""
import json
from pathlib import Path
from uuid import NAMESPACE_URL,uuid5

from scripts.plan_available_feature_providers import ROOT,sha,plan as feature_plan
from scripts.plan_nasa_first_lifecycle_contracts import plan as contract_plan


def evidence():
    directory=ROOT/'docs/reviews'
    names=['competition_nasa_first_lifecycle_contracts.json','competition_nasa_first_lifecycle_runtime.json',
           'competition_nasa_first_native_training_graph.json','competition_nasa_first_raw_training_graph.json']
    reports={name:json.loads((directory/name).read_text()) for name in names}
    if not all(report['passed'] for report in reports.values()):
        raise ValueError('Current lifecycle qualification required')
    runtime=reports[names[1]]
    if runtime['validator_sha256']!=sha(ROOT/'scripts/validate_nasa_first_lifecycle_runtime.py'):
        raise ValueError('Runtime validator changed')
    dependencies=runtime['dependencies']
    if dependencies['reviewer_sha256']!=sha(ROOT/'scripts/review_lifecycle_runtime_dependencies.py'):
        raise ValueError('Dependency reviewer changed')
    for package,relative in [('sciona',ROOT/'pyproject.toml'),('sciona-atoms-ml',ROOT.parent/'sciona-atoms-ml/pyproject.toml')]:
        if dependencies['source_manifest_sha256'][package]!=sha(relative):raise ValueError('Manifest changed')
    if dependencies['profile_sha256']!=sha(ROOT/'requirements/lifecycle-cpu.txt'):raise ValueError('Runtime profile changed')
    for package in dependencies['additional_package_notices'].values():
        for notice in package['retained_notices']:
            if sha(ROOT/notice['retained_notice'])!=notice['sha256']:raise ValueError('Retained notice changed')
    for name,digest in reports[names[2]]['source_closure_sha256'].items():
        repo,relative=name.split('/',1)
        base=ROOT if repo=='sciona-matcher' else ROOT.parent/repo/'src'
        if sha(base/relative)!=digest:raise ValueError('Native qualification source changed: '+name)
    for name,digest in runtime['execution']['implementation_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Policy execution source changed')
    return dict(reports={name:sha(directory/name) for name in names},
                planner_sha256=sha(Path(__file__)))


def plan():
    reviewed=evidence()
    candidates=contract_plan()
    frozen=json.loads((ROOT/'docs/reviews/competition_nasa_first_lifecycle_contracts.json').read_text())
    if candidates!=frozen:raise ValueError('Candidate contracts changed')
    atoms=[]
    for provider in candidates['providers']:
        atoms.append(dict(provider,artifact_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+provider['fqdn']))))
    return dict(approved=False,catalog_mutations=0,trust_tier_candidate=3,atoms=atoms,
        source_provenance=feature_plan()['source_provenance'],evidence=reviewed,
        scope='Non-publishable lifecycle provider drafts; original intake remains unchanged.')


if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/competition_nasa_first_lifecycle_provider_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['atoms']),catalog_mutations=0,approved=False)))
