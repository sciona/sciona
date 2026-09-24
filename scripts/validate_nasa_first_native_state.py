"""Verify portable bank state with all graph-trained synthetic native models."""
import argparse
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sciona.catboost_bank_state import pack_model_bank,unpack_model_bank
from scripts.validate_nasa_first_model_handoff import ROOT,sha


def validate(checkpoint):
    qualification=json.loads((checkpoint/'qualification.json').read_text())
    report_path=ROOT/'docs/reviews/competition_nasa_first_native_training_graph.json'
    if not qualification['passed'] or qualification!=json.loads(report_path.read_text()) or qualification['independent_native_fits']!=21:
        raise ValueError('Complete graph-native qualification required')
    models={}
    for fit in qualification['fits']:
        index=fit['model_index'];path=checkpoint/f'model_{index}.cbm'
        if sha(path)!=fit['model_sha256']:raise ValueError('Graph-native model integrity differs')
        model=CatBoostRegressor(thread_count=1);model.load_model(str(path));models[index]=model
    if set(models)!=set(range(21)):raise ValueError('All independent native models required')
    # Opaque application labels demonstrate that serialization does not require source-domain identifiers.
    bank={f'population-{index}':{0:models[2*index],1:models[2*index+1],2:models[2*index],
                              'global_model':models[20]} for index in range(10)}
    encoded=pack_model_bank(bank)
    restored=unpack_model_bank(json.loads(json.dumps(encoded,allow_nan=False)))
    if len(encoded['payload']['objects'])!=21 or len({id(model) for slots in restored.values() for model in slots.values()})!=21:
        raise ValueError('Model-bank cardinality differs')
    comparisons=0;rng=np.random.default_rng(732)
    compared=set()
    for population,slots in bank.items():
        selected=restored[population]
        if selected[0] is not selected[2] or selected['global_model'] is not restored['population-0']['global_model']:
            raise ValueError('Portable state alias/sharing differs')
        for slot,model in slots.items():
            if id(model) in compared:continue
            compared.add(id(model));categories=set(model.get_cat_feature_indices())
            frame=pd.DataFrame({name:['synthetic-unseen']*32 if i in categories else rng.integers(0,100,32)
                                for i,name in enumerate(model.feature_names_)})
            np.testing.assert_array_equal(model.predict(frame,thread_count=1),selected[slot].predict(frame,thread_count=1))
            comparisons+=len(frame)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,native_threads=1,
        independent_native_models=21,exact_prediction_comparisons=comparisons,populations=10,
        integer_slots_preserved=True,residual_alias_pairs=10,global_model_shared=True,json_roundtrip=True,
        qualification_sha256=sha(report_path),
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/catboost_model_state.py',
            'sciona/catboost_bank_state.py','scripts/validate_nasa_first_native_state.py',
            'tests/test_catboost_model_state.py','tests/test_catboost_bank_state.py']},
        provider_sha256=sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/model_selection/native_model_state.py'),
        limitations=['Payloads remain runtime material; no native models or training records are embedded in this report.',
            'Exact backend version required. Digests verify integrity, not external authenticity.',
            'Model bank persistence is qualified independently of inference policy persistence and catalog publication.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-directory',type=Path,required=True)
    report=validate(parser.parse_args().checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_native_state.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,models=21,exact_prediction_comparisons=report['exact_prediction_comparisons'])))
