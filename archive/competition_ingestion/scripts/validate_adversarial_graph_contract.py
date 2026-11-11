"""Structural serialized graph and provider witness checks, not neural execution."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_graph import build_adversarial_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.atoms.dl.adversarial_execution as provider


def main():
    digest,nodes,edges=encode_execution_graph(build_adversarial_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert len(graph.nodes)==2 and len(graph.edges)==1
    values={'payload':{}}
    for node in graph.nodes:
        fn=getattr(provider,node.matched_primitive.rsplit('.',1)[-1])
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs]
        values[node.outputs[0].name]=getattr(provider,'witness_'+fn.__name__)(**{p.name:values[p.name] for p in node.inputs})
    assert values['result']=={'kind':'Adversarial.Result'}
    try:provider.witness_adversarial_execute({'kind':'wrong'})
    except ValueError:pass
    else:raise AssertionError('wrong witness accepted')
    paths=['sciona/adversarial_graph.py','scripts/validate_adversarial_graph_contract.py']
    report={'format':'adversarial-graph-contract.v1','result':'passed','approved':False,
            'checks':{'serialized_nodes':2,'serialized_edges':1,'provider_witnesses':2,'wrong_witness_rejected':True},
            'graph_sha256':digest,'provider_sha256':hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'limits':'Structural codec and witness validation only. No real runner execution, serving or publication claim.'}
    (ROOT/'docs/reviews/competition_adversarial_graph_contract.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
