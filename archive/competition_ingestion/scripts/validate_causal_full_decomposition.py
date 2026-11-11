"""Run the pinned-source oracle on the complete explicit feature graph."""
import argparse
import asyncio
import hashlib
import inspect
import json
from pathlib import Path

from sciona.causal_decomposed_features import build_explicit_causal_feature_graph
from sciona.atoms.numerical.statistics import empirical_features
from scripts import validate_causal_decomposed_parity as oracle


async def validate(reference_dir):
    previous=oracle.build_explicit_causal_prediction_graph
    oracle.build_explicit_causal_prediction_graph=build_explicit_causal_feature_graph
    try:
        report=await oracle.validate(reference_dir)
    finally:
        oracle.build_explicit_causal_prediction_graph=previous
    root=Path(__file__).resolve().parents[1]
    for name in ['scripts/validate_causal_full_decomposition.py','sciona/causal_decomposed_features.py',
                 'tests/test_causal_decomposed_features.py']:
        report['implementation_hashes'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    report['provider_hashes'][empirical_features.__name__]=hashlib.sha256(Path(inspect.getfile(empirical_features)).read_bytes()).hexdigest()
    report['limitations']=[
        'No catalog promotion performed; provider/interface/dependency review remains required',
        'Synthetic same-library parity; no historical-library reproduction, real-data accuracy or causal-validity claim',
        'One-statistic batch adapter preserves explicit scalar conventions; reusable mode contracts still require publication review',
    ]
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=asyncio.run(validate(args.reference_dir))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'nodes':report['nodes'],'cases':report['cases']},indent=2))
