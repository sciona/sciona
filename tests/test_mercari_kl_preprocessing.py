import json
import pandas as pd
import pytest
from sciona.mercari_table_preprocessing import (
    MercariFillEmpty, MercariKLPreprocessor, export_kl_preprocessor, restore_kl_preprocessor,
)


def frame():
    # Synthetic values; names below are the public source-code interface.
    return pd.DataFrame(dict(name=['Synthetic A', None, 'Synthetic B'],
        item_description=['16gb!!!', 'No description yet', None],
        brand_name=['maker', 'maker', 'rare'], category_name=['Women/Test/Sub', 'A', None],
        shipping=[1, None, 0], item_condition_id=[1, None, 2]), index=[3, 1, 8])


def test_source_policy_and_input_isolation():
    raw = frame()
    filled = MercariFillEmpty().transform(raw)
    assert pd.isna(raw.loc[1, 'name']) and filled.loc[1, 'name'] == 'unk'
    model = MercariKLPreprocessor(1, [(r'(\d+)([a-z]+)', r'\1 \2'), ('!!+', '!!')])
    output = model.fit_transform(filled)
    assert output['brand_name'].tolist() == ['maker', 'maker', 'Other']
    assert output['item_description'].tolist() == ['16 gb!!', '', 'unk']
    assert output['category_name_l1s'].tolist() == ['Women/Test', 'A', 'unk']
    assert output['no_description'].tolist() == ['False', 'True', 'False']
    assert list(output.index) == [3, 1, 8] and 'no_description' not in filled


def test_json_prediction_replay_and_failed_refit():
    data = MercariFillEmpty().transform(frame())
    model = MercariKLPreprocessor(1, []).fit(data)
    restored = restore_kl_preprocessor(json.loads(json.dumps(export_kl_preprocessor(model))))
    pd.testing.assert_frame_equal(restored.transform(data), model.transform(data))
    query = data.copy(); query['brand_name'] = 'unseen'
    assert set(restored.transform(query)['brand_name']) == {'Other'}
    invalid = data.copy(); invalid['brand_name'] = 1
    with pytest.raises(ValueError): model.fit(invalid)
    with pytest.raises(ValueError): model.transform(data)


def test_whitelist_precedes_trimming():
    data = MercariFillEmpty().transform(frame())
    data['brand_name'] = 'x' * 101
    model = MercariKLPreprocessor(1, []).fit(data)
    assert set(model.transform(data)['brand_name']) == {'Other'}


@pytest.mark.parametrize('key,value', [('popular', ['x', 'x']), ('popular', [1]),
    ('num_brands', True), ('pandas_version', 'unknown'), ('extra', True)])
def test_invalid_state(key, value):
    model = MercariKLPreprocessor(1, []).fit(MercariFillEmpty().transform(frame()))
    state = export_kl_preprocessor(model); state[key] = value
    with pytest.raises(ValueError): restore_kl_preprocessor(state)
