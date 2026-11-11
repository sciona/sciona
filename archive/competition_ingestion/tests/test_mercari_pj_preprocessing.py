import json
import pandas as pd
import pytest
from sciona.mercari_table_preprocessing import MercariFillEmpty
from sciona.mercari_pj_preprocessing import (
    MercariPJPreprocessor, export_pj_preprocessor, restore_pj_preprocessor,
)


def frame():
    # Public source-code interface with entirely synthetic values.
    return pd.DataFrame(dict(name=['Café running :)', 'bbaa', None],
        item_description=['No description yet', 'synthetic bundle', 'Bundle'],
        brand_name=['maker', 'maker', None], category_name=['A/B/C', 'A', None],
        shipping=[1, 0, 1], item_condition_id=[1, None, 2]), index=[4, 2, 7])


def test_explicit_source_policy_and_input_isolation():
    data = MercariFillEmpty().transform(frame())
    original = data.copy(deep=True)
    output = MercariPJPreprocessor().transform(data)
    assert output['name_clean'].tolist() == ['cafe run smiley', 'bbaa', 'unk']
    assert output['desc_clean'].iloc[0] == 'unk'
    assert output['is_bundle'].tolist() == [0, 1, 0]
    assert output.loc[4, 'cat_2'] == 'b' and pd.isna(output.loc[2, 'cat_2'])
    pd.testing.assert_frame_equal(data, original)
    assert list(output.index) == [4, 2, 7]


@pytest.mark.parametrize('stem', [False, True])
@pytest.mark.parametrize('hashchars', [False, True])
def test_modes_and_json_restore(stem, hashchars):
    data = MercariFillEmpty().transform(frame())
    model = MercariPJPreprocessor(stem=stem, hashchars=hashchars)
    restored = restore_pj_preprocessor(json.loads(json.dumps(export_pj_preprocessor(model))))
    pd.testing.assert_frame_equal(restored.transform(data), model.transform(data))
    assert ('name_clean' in restored.transform(data)) == stem
    if stem and hashchars:
        assert restored.transform(data).loc[2, 'name_clean'] == 'ab'


@pytest.mark.parametrize('params', [dict(n_jobs=4), dict(n_jobs=True), dict(stem=1), dict(hashchars='false')])
def test_invalid_resource_or_policy_controls(params):
    with pytest.raises(ValueError): MercariPJPreprocessor(**params).transform(frame())
