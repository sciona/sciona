"""Execute the full independent Amex pipeline from synthetic raw JSON inputs."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amex_synthetic import payload
from sciona.amex_execution import prepare,execute


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    paths=sorted((ROOT/'sciona').glob('amex_*.py'))+[ROOT/'scripts/amex_synthetic.py',Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    result=execute(prepare(payload()))
    assert result['models']==25 and result['model_counts']==dict(row=5,downstream_tree=10,neural=10)
    assert len(result['scores'])==8 and all(0<=v<=.900000000000001 for v in result['scores'])
    assert result['score_kind']=='literal_weighted_sum_0.9';json.dumps(result,allow_nan=False)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,models=25,query_customers=8,raw_preprocessing=True,full_training_controls=True,strict_json=True,sha256=before,
        limits=['Explicit synthetic raw schema roles and generalized feature widths; historical volume, accuracy and native/GPU parity unqualified.','Serialized graph and publication gates remain pending.'])
    (ROOT/'docs/reviews/competition_amex_raw_execution.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('sha256','limits')}),flush=True)


if __name__=='__main__':main()
