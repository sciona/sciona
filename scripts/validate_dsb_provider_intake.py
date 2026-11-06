"""Reject corrupted DSB draft plans before any database writes."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import import_dsb_provider_drafts as intake


def validate():
    original=intake.reviewed_plan()
    mutations={
        'graph_digest':lambda p:p.update(graph_digest='invalid'),
        'execution_evidence':lambda p:p.update(execution_evidence_sha256='invalid'),
        'interface_evidence':lambda p:p.update(interface_plan_sha256='invalid'),
        'planner':lambda p:p.update(planner_sha256='invalid'),
        'identity':lambda p:p['new_providers'][0].update(atom_id='invalid'),
        'version':lambda p:p['new_providers'][0].update(version_id='invalid'),
        'content':lambda p:p['new_providers'][0].update(content_hash='invalid'),
        'source':lambda p:p['new_providers'][0].update(provider_sha256='invalid'),
        'fingerprint':lambda p:p['new_providers'][0].update(fingerprint={}),
        'seed_fields':lambda p:p['new_providers'][0]['seed_fields'].update(description='invalid'),
        'ports':lambda p:p['new_providers'][0].update(interfaces=[]),
        'owner':lambda p:p['new_providers'][0].update(owner={}),
        'missing_provider':lambda p:p['new_providers'].pop(),
        'duplicate_provider':lambda p:p['new_providers'].__setitem__(0,p['new_providers'][1]),
    }
    rejected=[]
    with TemporaryDirectory(prefix='dsb-intake-contract-') as directory:
        path=Path(directory)/'plan.json'
        for name,mutate in mutations.items():
            altered=deepcopy(original); mutate(altered)
            path.write_text(json.dumps(altered))
            with patch.object(intake,'PLAN',path):
                try:intake.reviewed_plan()
                except ValueError:rejected.append(name)
                else:raise AssertionError('Corruption accepted: '+name)
    return dict(passed=True,catalog_mutations=0,rejected_corruptions=rejected,
                plan_sha256=intake.sha(intake.PLAN),importer_sha256=intake.sha(Path(intake.__file__)),
                validator_sha256=intake.sha(Path(__file__)))


if __name__=='__main__':
    report=validate()
    (intake.ROOT/'docs/reviews/dsb_provider_intake_corruption.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
