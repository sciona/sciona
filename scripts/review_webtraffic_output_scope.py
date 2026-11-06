"""Compare generic output alignment with pinned final-route source on synthetic panels."""
import argparse
import ast
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.panel_output import aligned_suffix_records
from scripts.audit_decomposed_webtraffic_readiness import ROOT,sha,ORIGINAL,SOURCE_HASH
from scripts.import_flavours_provider_drafts import connect
from sciona.competition_import import _canonical
import hashlib


def review(source):
    pin=json.loads((ROOT/'docs/reviews/competition_webtraffic_notebook_pin.json').read_text())
    code=(source/'submission-final-code.py').read_text()
    if hashlib.sha256(code.encode()).hexdigest()!=pin['code_sha256']:raise ValueError('Pinned final-route code required')
    tree=ast.parse('\n'.join(line for line in code.splitlines() if not line.startswith('%')))
    selected=[];active=False
    for node in tree.body:
        name=node.targets[0].id if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) else None
        if name=='subm_preds':
            active=True;node=copy.deepcopy(node)
            # Replace only the historical calendar boundary with a caller-supplied
            # synthetic coordinate. Preserve suffix selection, assertions and flattening.
            node.value.slice.elts[1].lower=ast.Name(id='cutoff',ctx=ast.Load())
        if active:selected.append(node)
        if name=='answers':break
    if len(selected)!=4 or not all(isinstance(n,ast.Assert) for n in selected[1:3]):
        raise ValueError('Exact suffix selection, two alignment assertions and output assembly required')
    program=compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'pinned-source-output','exec')
    checks=0
    for rows in (1,3,7):
        for width in (1,4,9):
            for suffix in sorted({1,width}):
                r=[f'entity-{i}' for i in range(rows)];c=[f'axis-{i}' for i in range(width)]
                values=(np.arange(rows*width,dtype=np.int64).reshape(rows,width)*3)-20
                ids=np.arange(rows*suffix,dtype=np.int64).reshape(rows,suffix)+100
                env=dict(np=np,pd=pd,f_preds=pd.DataFrame(values,index=r,columns=c),
                    keys=pd.DataFrame(ids,index=r,columns=c[-suffix:]),cutoff=c[-suffix])
                exec(program,env)
                actual_ids,actual_values=aligned_suffix_records(values,r,c,r,c[-suffix:],ids)
                expected=env['answers'].to_numpy()
                np.testing.assert_array_equal(actual_ids,expected[:,0])
                np.testing.assert_array_equal(actual_values,expected[:,1]);checks+=1
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
        if len(rows)!=1 or hashlib.sha256(_canonical(rows[0]['details']['snapshot']).encode()).hexdigest()!=SOURCE_HASH:
            raise ValueError('Immutable original source drift')
        template=rows[0]['details']['snapshot']['template']
        stages=[s['stage_id'] for s in template['stages']]
        if len(stages)!=5 or template['outputs'][0]['type_desc']!='SubmissionArtifact':
            raise ValueError('Original final-output contract changed')
    return dict(format='webtraffic-original-output-scope.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,source_version_id=ORIGINAL,source_content_hash=SOURCE_HASH,
        source_commit=pin['commit'],source_code_sha256=pin['code_sha256'],original_stages=stages,
        synthetic_source_comparisons=checks,
        decision='Original finalization must include an explicit output coordinate suffix and supplied identifier matrix. Preserve exact axis alignment and row-major flattening after model/checkpoint averaging, zero fill, threshold and integer rounding. Return identifiers and values in memory; file compression and historical fixed calendar boundaries are serialization and invocation policy.',
        reusable_operation='Domain-independent int64 panel suffix alignment and flattening, preserving arbitrary string axis labels and exact int64 values; no page parsing, date inference or forecast model dependency.',
        source_adaptation='The comparison changes only the historical calendar slice lower bound to a caller-provided synthetic coordinate. All remaining selected source expressions execute unchanged on synthetic frames.',
        implementation_sha256={p:sha(ROOT/p) for p in ['sciona/panel_output.py','tests/test_panel_output.py','scripts/review_webtraffic_output_scope.py']},
        limits=['New output operation is implemented and source-compared but not registered, connected to the decomposed graph or approved.',
                'Input predictions must already be finalized int64 values; numerical rounding and ensemble policy remain upstream.',
                'The current approved derivative continues to return its complete dated forecast panel unchanged.',
                'Runtime key parsing and graph-root output controls still require implementation and qualification. No dataset records, identifiers or historical calendar constants are retained in this report.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    result=review(parser.parse_args().source_root)
    (ROOT/'docs/reviews/webtraffic_original_output_scope.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,synthetic_source_comparisons=result['synthetic_source_comparisons'],approved=False)))
