"""Tabular Ensemble publication must reject review/evidence/dependency drift before writes."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import promote_tabular_ensemble_execution as publication


def main():
    publication.review(ROOT)
    original=Path.read_text
    rejected=[]
    cases=[('semantic_review','proposed_tier',1),('semantic_review','verdict','pending'),
           ('semantic_review','review_source','human'),('graph_execution','serialized_graph_sha256','invalid')]
    for suffix,field,value in cases:
        target=ROOT/f'docs/reviews/competition_tabular_ensemble_{suffix}.json'
        def read(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path==target:
                document=json.loads(text);document[field]=value;return json.dumps(document)
            return text
        with patch.object(Path,'read_text',read):
            try:publication.review(ROOT)
            except ValueError:rejected.append(suffix+':'+field)
            else:raise AssertionError('incompatible review accepted')
    real_sha=publication.sha
    with patch.object(publication,'sha',side_effect=lambda p:'invalid' if p.name=='competition_tabular_ensemble_environment.json' else real_sha(p)):
        try:publication.review(ROOT)
        except ValueError:rejected.append('evidence_hash')
        else:raise AssertionError('stale evidence accepted')
    with patch('importlib.metadata.version',return_value='invalid'):
        try:publication.review(ROOT)
        except ValueError:rejected.append('installed_dependency')
        else:raise AssertionError('changed dependency accepted')
    report={'format':'tabular_ensemble-publication-gates.v1','result':'passed','rejected_gates':rejected,
            'publisher_sha256':real_sha(ROOT/'scripts/promote_tabular_ensemble_execution.py'),
            'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits':'Six injected incompatible inputs rejected by read-only review; no catalog mutations.'}
    (ROOT/'docs/reviews/competition_tabular_ensemble_publication_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rejected_gates':len(rejected),'result':'passed'}))


if __name__=='__main__':main()
