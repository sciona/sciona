"""Domain feature state paired with a reusable residual-classifier model bundle."""
from sciona.nasa_workflow_inputs import prepare_prediction_inputs,attach_predictions,ETD_FEATURES,HISTORY_FEATURES
from sciona.residual_model_state import validate_state,predict_state
from sciona.tabular_contracts import stable_feature_order


def validate_adapter_state(state):
    validate_state(state)
    adapter=state['adapter_state']
    if set(adapter)!={'format','airport','vocabulary','target_unit'} or adapter['format']!='sciona.airport.feature-state.v1' or adapter['target_unit']!='minutes':
        raise ValueError('Complete airport adapter state in minutes required')
    vocabulary=adapter['vocabulary']
    if not isinstance(vocabulary,list) or any(not isinstance(value,str) or len(value)!=3 for value in vocabulary) or len(set(vocabulary))!=len(vocabulary):
        raise ValueError('Persisted training vocabulary required')
    names=stable_feature_order(['unix_time']+ETD_FEATURES+['Other']+vocabulary+HISTORY_FEATURES)
    if names!=state['feature_names']:
        raise ValueError('Adapter vocabulary and model feature schema differ')
    if not isinstance(adapter['airport'],str) or len(adapter['airport'])!=4:
        raise ValueError('Persisted four-character airport required')
    return adapter,names,tuple(vocabulary)


def predict_records(prediction_records,state):
    adapter,names,vocabulary=validate_adapter_state(state)
    features,identities=prepare_prediction_inputs(prediction_records,adapter['airport'],names,tuple(vocabulary))
    return attach_predictions(identities,predict_state(features,state))
