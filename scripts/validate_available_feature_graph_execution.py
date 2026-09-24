"""Execute codec-roundtripped graphs built from exact staged provider ports."""
import asyncio
import contextlib
import inspect
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import sciona.atoms.ml.tabular.available_features as provider
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.available_feature_graph_cases import cases
from scripts.plan_available_feature_providers import ROOT,plan,sha
from scripts.validate_available_feature_database_gates import check_staged
from scripts.validate_available_feature_providers import same


def validate():
    proposed=plan();fixtures=cases();graphs=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed)
        for atom in proposed['atoms']:
            ports=db.execute('SELECT direction,name,type_desc,constraints,required,default_value_repr,dim_signature FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                (atom['version_id'],)).fetchall()
            def selected(direction):
                return [IOSpec(**{key:value for key,value in port.items() if key!='direction'}) for port in ports if port['direction']==direction]
            node=AlgorithmicNode(node_id='operation',name=atom['runtime_fqdn'].split('.')[-1],description=atom['description'],
                concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=atom['runtime_fqdn'],inputs=selected('input'),outputs=selected('output'))
            graph=CDGExport(nodes=[node],edges=[],metadata=dict(artifact_source='materialized_provider_qualification',
                provider_version_id=atom['version_id'],provider_content_hash=atom['content_hash']))
            digest,nodes,edges=encode_execution_graph(graph)
            restored=decode_execution_graph(nodes,edges,digest)
            if restored!=graph:raise ValueError('Provider graph codec roundtrip differs')
            graphs.append((atom,restored,digest))
    outcomes=[]
    with tempfile.TemporaryDirectory(prefix='sciona-feature-graphs-') as directory:
        for atom,graph,digest in graphs:
            name=graph.nodes[0].name;target,args,kwargs=fixtures[name]
            expected=getattr(provider,target)(*args,**kwargs)
            inputs=inspect.signature(getattr(provider,name)).bind(*args,**kwargs).arguments
            captured={}
            def capture(path,node,port,value):
                if port.startswith('out_'):captured[port[4:]]=value
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture),\
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-available-features',name).execute(dict(inputs),cdg=graph))
            if result['status']!='completed':raise ValueError('Materialized provider graph failed')
            ports=graph.nodes[0].outputs
            values=list(expected) if len(ports)>1 else [expected]
            if len(values)!=len(ports) or set(captured)!={port.name for port in ports}:raise ValueError('Provider output arity differs')
            for port,value in zip(ports,values):same(captured[port.name],value)
            outcomes.append(dict(runtime_fqdn=atom['runtime_fqdn'],version_id=atom['version_id'],
                provider_content_hash=atom['content_hash'],graph_sha256=digest,outputs_checked=len(ports),passed=True))
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,provider_graphs=len(outcomes),
        stored_ports_used=True,codec_roundtrips=True,production_executor=True,cases=outcomes,
        implementation_sha256={name:sha(ROOT/name) for name in ['scripts/validate_available_feature_graph_execution.py',
            'scripts/available_feature_graph_cases.py','scripts/validate_available_feature_database_gates.py',
            'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']},
        limitations=['Single-provider execution graphs are constructed from staged ports; this is not complete original-workflow CDG execution.',
            'Intermediate-value persistence is replaced by output capture; computation and graph dispatch are the production executor.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/available_feature_provider_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,provider_graphs=report['provider_graphs'],outputs_checked=sum(c['outputs_checked'] for c in report['cases']),approved=False)))
