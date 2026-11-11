"""Execute the full 220-model synthetic lifecycle through private JSON."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validate_m5_full_inventory import configuration
from sciona.m5_execution import prepare,execute


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    paths=sorted((ROOT/'sciona').glob('m5_*.py'))+[Path(__file__).resolve(),ROOT/'scripts/validate_m5_full_inventory.py']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    config=json.loads(json.dumps(configuration(),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v))
    payload=dict(version=1,identities=['synthetic-'+str(i) for i in range(70)],configuration=config,
                 controls=dict(recursive_first_day=0,nonrecursive_first_day=710))
    print('Executing private synthetic JSON boundary: 220 models and 28 forecast days',flush=True)
    result=execute(prepare(payload))
    assert result['models']==220 and result['horizon']==28 and result['model_families']==6
    assert np.asarray(result['forecast']).shape==(70,28)
    encoded=json.dumps(result,allow_nan=False)
    assert not any(identity in encoded for identity in payload['identities'])
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,
        checks=dict(synthetic_only=True,models=220,horizon=28,synthetic_series=70,strict_json_output=True,
                    identifiers_excluded=True,code_unchanged=True),sha256=before,
        scope='Full model inventory and horizon through private boundary on reduced synthetic population; no historical accuracy qualification.')
    (ROOT/'docs/reviews/competition_m5_boundary_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
