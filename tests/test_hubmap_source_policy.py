"""Synthetic AST checks for selecting model policy without unrelated values."""
import pytest
from scripts.review_hubmap_source_policy import extract_config


def test_top_level_patience_and_allowlisted_values_only():
    source="config={'FOLD_LIST':[0,1,2,3],'patience':50,'scheduler':{'patience':5},'unrelated':unavailable()}"
    assert extract_config(source,['FOLD_LIST','patience'])==dict(FOLD_LIST=[0,1,2,3],patience=50)


@pytest.mark.parametrize('source',["config={}","a={'FOLD_LIST':[0]}; b={'FOLD_LIST':[1]}"])
def test_missing_or_ambiguous_policy_rejected(source):
    with pytest.raises(ValueError):extract_config(source,['FOLD_LIST'])


def test_policy_expression_is_not_executed():
    with pytest.raises(ValueError):extract_config("config={'FOLD_LIST':unknown()}",['FOLD_LIST'])
