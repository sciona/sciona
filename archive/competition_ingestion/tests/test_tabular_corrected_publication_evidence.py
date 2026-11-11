"""Reject weakened or changed stored-execution qualification."""
from copy import deepcopy
import json
import pytest
from scripts import tabular_publication_evidence as evidence


def reports():
    return [json.loads((evidence.ROOT/path).read_text()) for path in (evidence.EXECUTION,evidence.SOURCE)]


def test_exact_reports():
    execution,source=reports()
    evidence.validate_reports(execution,source,deepcopy(execution),deepcopy(source))


@pytest.mark.parametrize('field,value',[
    ('passed',1),('catalog_mutations',False),('stored_catalog_graph_executed',False),
    ('graph_sha256','0'*64),('version_id','wrong'),('artifact_id','wrong'),
    ('tests',{}),('software_sha256',{}),
])
def test_changed_execution(field,value):
    execution,source=reports();changed=deepcopy(execution);changed[field]=value
    with pytest.raises(ValueError):evidence.validate_reports(changed,source,execution,source)


def test_changed_scope():
    execution,source=reports();changed=deepcopy(source);changed['source_stage_review'].pop()
    with pytest.raises(ValueError):evidence.validate_reports(execution,changed,execution,source)
