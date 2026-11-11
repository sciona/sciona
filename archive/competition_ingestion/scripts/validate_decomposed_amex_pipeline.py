"""Compare all decomposed learner stages with the full qualified CPU recipe."""
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts.amex_synthetic import payload
from sciona.amex_execution import prepare,execute
from sciona import amex_pipeline as reference
from sciona import decomposed_amex_pipeline as candidate
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    paths=sorted((ROOT/'sciona').glob('amex_*.py'))+[
        ROOT/'sciona/decomposed_amex_pipeline.py',ROOT/'scripts/amex_synthetic.py',Path(__file__).resolve(),
        Path(inspect.getfile(ordered_weighted_sum))]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):sha(p) for p in paths}
    p=payload();serialized=json.dumps(p,sort_keys=True);prepared=prepare(p);captured={}
    original_blend=reference.blend
    def capture(*values):
        captured['branches']=np.stack(values).copy()
        return original_blend(*values)
    rng=torch.get_rng_state().clone()
    print('Running complete reference: 15 DART models and 10 neural models',flush=True)
    with patch.object(reference,'blend',side_effect=capture):expected=execute(prepared)
    print('Running complete decomposed recipe',flush=True)
    population=candidate.populations(prepared);features=candidate.features(population)
    row=candidate.row_models(population,features)
    print('Five row models complete',flush=True)
    assembled=candidate.assembly(features,row)
    trees=candidate.tree_models(population,assembled)
    print('Ten downstream tree models complete',flush=True)
    neural=candidate.neural_models(population,features,assembled)
    print('Ten neural models complete',flush=True)
    values,weights=candidate.combination_inputs(trees,neural)
    np.testing.assert_array_equal(values,captured['branches'])
    scores=ordered_weighted_sum(values,weights)
    np.testing.assert_array_equal(scores,original_blend(*values))
    assert candidate.result(scores)==expected
    assert json.dumps(p,sort_keys=True)==serialized and torch.equal(rng,torch.get_rng_state())
    assert before=={str(path):sha(path) for path in paths}
    report=dict(format='decomposed-amex-pipeline.v1',passed=True,approved=False,catalog_mutations=0,
        complete_reference_models=25,complete_candidate_models=25,exact_branch_probabilities=True,
        exact_final_output=True,input_and_torch_rng_preserved=True,
        implementation_sha256={str(path.relative_to(ROOT)):sha(path) for path in paths if path.is_relative_to(ROOT)},
        generic_provider_sha256=sha(Path(inspect.getfile(ordered_weighted_sum))),
        limits='Synthetic full CPU training; no historical accuracy or GPU equivalence. Registered graph execution, provider qualification and original publication remain pending.')
    (ROOT/'docs/reviews/decomposed_amex_pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}),flush=True)


if __name__=='__main__':main()
