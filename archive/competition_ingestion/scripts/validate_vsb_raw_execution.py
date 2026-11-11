"""Execute complete independent preprocessing and ensemble on synthetic raw signals."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vsb_synthetic import payload
from sciona.vsb_execution import prepare,execute


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    paths=sorted((ROOT/'sciona').glob('vsb_*.py'))+[ROOT/'scripts/vsb_synthetic.py',Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    result=execute(prepare(payload()))
    assert result['models']==125 and len(result['probabilities'])==8
    assert all(0<=p<=1 for p in result['probabilities'])
    assert result['signal_decisions']==[[int(p>result['threshold'])]*3 for p in result['probabilities']]
    json.dumps(result,allow_nan=False)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,models=125,query_measurements=8,query_signal_decisions=24,raw_preprocessing=True,full_training_limits=True,strict_json=True,sha256=before,limits=['Synthetic raw signals at generalized input length; serialized graph and historical precision/accuracy remain unqualified.'])
    (ROOT/'docs/reviews/competition_vsb_raw_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sha256','limits')}))


if __name__=='__main__':main()
