"""Replay frozen feature and routing contract checks against relocated providers."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

import pandas as pd
from sciona._runtime.ml import osic_quantile_features
from scripts.audit_osic_training_finalization import ROOT, REVISION


def frozen(name):
    return subprocess.check_output(['git','show',REVISION+':'+name],cwd=ROOT)


def qualify():
    qualified = json.loads(subprocess.check_output(['git','show',
        'f28f2dde:docs/reviews/osic_relocated_combined_graph.json'],cwd=ROOT))
    def verify_software():
        for name,digest in qualified['current_software_sha256'].items():
            if hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Numerically qualified software changed')
    verify_software()
    fixture_source = frozen('scripts/validate_osic_quantile_features.py')
    fixture = next(n for n in ast.parse(fixture_source).body
                   if isinstance(n,ast.FunctionDef) and n.name=='synthetic_tables')
    fixture_module = ModuleType('scripts.validate_osic_quantile_features')
    fixture_module.__dict__.update(pd=pd,FIELDS=osic_quantile_features.FIELDS)
    exec(compile(ast.Module(body=[fixture],type_ignores=[]),'<frozen-synthetic-tables>','exec'),fixture_module.__dict__)
    reports=[]
    resolutions={}
    def resolved_sha(path):
        path=Path(path)
        try:
            relative=path.relative_to(ROOT)
        except ValueError:
            relative=None
        if relative and relative.parts[0]=='scripts':
            raw=frozen(str(relative))
            label='git:'+REVISION+':'+str(relative)
        elif relative and relative.parts[0]=='sciona' and len(relative.parts)==2:
            relocated=ROOT.parent/'sciona-atoms-ml/src/sciona/_runtime/ml'/relative.name
            raw=relocated.read_bytes()
            label=str(relocated.relative_to(ROOT.parent))
        else:
            raw=path.read_bytes()
            label=str(path.relative_to(ROOT.parent))
        digest=hashlib.sha256(raw).hexdigest()
        if label in resolutions and resolutions[label]!=digest:
            raise ValueError('Contract software changed during validation')
        resolutions[label]=digest
        return digest
    for script,reference in [
            ('validate_osic_feature_replay_provider','osic_feature_replay_provider'),
            ('validate_osic_image_routing','osic_image_routing')]:
        source=frozen('scripts/'+script+'.py')
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.If)
            and ast.unparse(n.test)=="__name__ == '__main__'")]
        changed=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module=='sciona':
                if not all(alias.name.startswith('osic_') for alias in node.names):
                    raise ValueError('Unexpected runtime imports')
                changed.extend(alias.name for alias in node.names)
                node.module='sciona._runtime.ml'
        expected={'validate_osic_feature_replay_provider':['osic_feature_state','osic_quantile_features'],
                  'validate_osic_image_routing':['osic_image_inference']}[script]
        if changed!=expected:
            raise ValueError('Unexpected frozen validator import contract')
        scope=dict(__file__=str(ROOT/'scripts'/(script+'.py')),__name__='frozen_contract_validator')
        with patch.dict(sys.modules,{'scripts.validate_osic_quantile_features':fixture_module}):
            exec(compile(tree,'<frozen-'+script+'>','exec'),scope)
        scope['sha']=resolved_sha
        result=scope['validate']()
        historical=json.loads(frozen('docs/reviews/'+reference+'.json'))
        for field in ('cases','rejected_contracts'):
            if result[field]!=historical[field]:
                raise ValueError('Frozen contract outcomes changed')
        reports.append(dict(validator=script,frozen_validator_sha256=hashlib.sha256(source).hexdigest(),
            relocated_imports=changed,cases=result['cases'],rejected_contracts=result['rejected_contracts'],
            current_provider_sha256=result['provider_sha256'],historical_contract_outcomes_reproduced=True))
    verify_software()
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        frozen_revision=REVISION,fixture_sha256=hashlib.sha256(fixture_source).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reports=reports,resolved_software_sha256=resolutions,
        limitations=[
            'Frozen feature replay and mapped population routing contracts only; no catalog mutation or approval renewal.',
            'Routing uses synthetic callbacks and local lifetime checks, not a decoder/model or global memory bound.',
            'These checks complement combined inference evidence; full training and original workflow remain outstanding.',
            'Provider/runtime relocation remains uncommitted and requires immutable version anchoring before publication.'])


if __name__=='__main__':
    result=qualify()
    (ROOT/'docs/reviews/osic_relocated_provider_contracts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['reports']),
        cases=sum(len(r['cases']) for r in result['reports']),
        rejections=sum(len(r['rejected_contracts']) for r in result['reports']))))
