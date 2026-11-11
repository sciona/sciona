"""Publication gate checks without modifying runtime evidence or catalog state."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.promote_trackml_execution import review

ROOT=Path(__file__).resolve().parents[1]


def test_review_accepts_current_connected_evidence():
    graph,digest,nodes,edges,specs,interfaces,boundary,signatures,evidence=review(ROOT)
    assert (len(nodes),len(edges),len(specs))==(3,2,3)
    assert [p.name for p in boundary['input']]==['payload']
    assert [p.name for p in boundary['output']]==['result']
    assert len(evidence)==16


@pytest.mark.parametrize('change', ['graph','provider','coverage','implementation'])
def test_review_rejects_invalid_evidence(change):
    original=Path.read_text
    def edited(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        if path.name=='competition_trackml_lifecycle.json':
            data=json.loads(text)
            if change=='graph':data['serialized_graph_sha256']='0'*64
            elif change=='provider':data['provider_sha256']='0'*64
            elif change=='coverage':data['checks']['serialized_graph_cases']=0
            else:data['implementation_sha256']['sciona/trackml_payload.py']='0'*64
            return json.dumps(data)
        return text
    with patch.object(Path,'read_text',edited),pytest.raises(ValueError):review(ROOT)
