"""Complete CHAMPS training and inference before explicit graph ensemble reduction.

Only SCIONA_CHAMPS_SOURCE_DIR selects the provisioned verified software cache.
All input geometry, checkpoint state and predictions are private runtime data.
"""
import gc
import os
from pathlib import Path
import tempfile

from sciona.champs_contract import Prepared,prepare
from sciona.champs_ensemble import MODEL_ORDER
from sciona.champs_lifecycle import fit_variant,load_for_prediction
from sciona.champs_population import population_batches,split_population
from sciona.champs_prediction import predict_batches
from sciona.champs_preprocessing import ChampsPreprocessor
from sciona.champs_source_runtime import ChampsSourceRuntime,EXECUTION_VERSION


def train_models(prepared):
    if type(prepared) is not Prepared:
        raise ValueError('Prepared CHAMPS population required')
    cache=os.environ.get('SCIONA_CHAMPS_SOURCE_DIR')
    if not cache:
        raise RuntimeError('Provision CHAMPS source through SCIONA_CHAMPS_SOURCE_DIR')
    runtime=ChampsSourceRuntime(Path(cache))
    preprocessor=ChampsPreprocessor(runtime)
    packed,state=preprocessor.prepare(prepared.training_atoms,prepared.training_couplings)
    if prepared.selection=='validation':
        train,validation=split_population(packed)
    else:
        train,validation=packed,None
    predictions,reports={},{}
    with tempfile.TemporaryDirectory(prefix='champs-private-runtime-') as directory:
        checkpoint=Path(directory)/'model-state.pt'
        for variant in MODEL_ORDER:
            epochs,options=prepared.model_options[variant]
            reports[variant]=fit_variant(runtime,variant,train,state,options,epochs=epochs,
                checkpoint_path=checkpoint,validation=validation)
            gc.collect()
            model,restored=load_for_prediction(runtime,variant,checkpoint)
            inference,_=preprocessor.prepare(prepared.inference_atoms,prepared.inference_couplings,state=restored,labeled=False)
            predictions[variant]=predict_batches(model,population_batches(inference,batch_size=64,
                shuffle=False,drop_last=False,seed=0),prepared.inference_keys)
            del model,restored,inference
            gc.collect()
    return {'version':1,'execution_version':EXECUTION_VERSION,
            'model_predictions':predictions,'coupling_types':prepared.inference_types,
            'models_completed':list(MODEL_ORDER),'training':reports}

