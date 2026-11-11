"""Compare synthetic execution envelopes across the shared graph-model change."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f28f2dde'
MODEL = 'sciona/architect/models.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(frozen_model=False):
    if frozen_model:
        raw = subprocess.check_output(['git', 'show', REVISION + ':' + MODEL], cwd=ROOT)
        module = types.ModuleType('sciona.architect.models')
        module.__file__ = '<reviewed-graph-model>'
        sys.modules[module.__name__] = module
        exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    from scripts.qualify_osic_relocated_combined_graph import stage_graphs
    from sciona.services.execution_graph_codec import encode_execution_graph
    with tempfile.TemporaryDirectory() as temporary:
        stage_graphs(Path(temporary))
        sys.path.insert(0, temporary)
        from frozen_osic_graphs.osic_combined_inference_graph import build_combined_inference_graph
        digest, nodes, edges = encode_execution_graph(build_combined_inference_graph())
        return dict(digest=digest, nodes=nodes, edges=edges)


def compare(old, new, path='', additions=None):
    """Require every old field/value and sequence order to remain exact."""
    if additions is None:
        additions = []
    if isinstance(old, dict):
        if not isinstance(new, dict) or not old.keys() <= new.keys():
            raise ValueError('Existing graph fields removed: ' + path)
        for key in old:
            compare(old[key], new[key], path + '/' + key, additions)
        for key in sorted(new.keys() - old.keys()):
            additions.append(dict(path=path + '/' + key, value=new[key]))
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            raise ValueError('Graph sequence changed: ' + path)
        for index, (a, b) in enumerate(zip(old, new)):
            compare(a, b, path + '/' + str(index), additions)
    elif type(old) is not type(new) or old != new:
        raise ValueError('Existing graph value changed: ' + path)
    return additions


def audit():
    paths = [ROOT / MODEL, ROOT / 'sciona/architect/handoff.py',
             ROOT / 'sciona/services/execution_graph_codec.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    cases = []
    for frozen in (True, False):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker']
        if frozen:
            command.append('--frozen-model')
        cases.append(json.loads(subprocess.check_output(command, cwd=ROOT, env=os.environ)))
    old, new = cases
    evidence = json.loads((ROOT / 'docs/reviews/osic_relocated_combined_graph.json').read_text())
    if old['digest'] != evidence['execution']['graph_sha256']:
        raise ValueError('Historical graph was not reproduced')
    snapshots = [json.loads(case['nodes'][0]['type_signature'])['graph'] for case in cases]
    additions = compare(*snapshots)
    if old['edges'] != new['edges']:
        raise ValueError('Edges changed')
    from sciona.services.execution_graph_codec import decode_execution_graph
    decode_execution_graph(new['nodes'], new['edges'], new['digest'])
    historical_error = None
    try:
        decode_execution_graph(old['nodes'], old['edges'], old['digest'])
    except ValueError as exc:
        historical_error = str(exc)
    rejected = 0
    for index in range(len(new['edges'])):
        try:
            decode_execution_graph(new['nodes'], new['edges'][:index] + new['edges'][index+1:], new['digest'])
        except ValueError:
            rejected += 1
        else:
            raise ValueError('Missing graph edge accepted')
    if before != {str(p.relative_to(ROOT)): sha(p) for p in paths}:
        raise ValueError('Graph software changed during audit')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
                auditor_sha256=sha(Path(__file__)),
                historical_revision=REVISION, historical_graph_sha256=old['digest'],
                current_graph_sha256=new['digest'], current_source_sha256=before,
                existing_fields_unchanged=True, added_fields=additions,
                nodes=len(new['nodes']), edges=len(new['edges']), missing_edges_rejected=rejected,
                historical_envelope_accepted=historical_error is None,
                historical_envelope_error=historical_error,
                limitations=['Structural comparison only; current-schema numerical execution still required.',
                             'No historical catalog envelope is rewritten or approved by this audit.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--frozen-model', action='store_true')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.frozen_model)))
    else:
        result = audit()
        (ROOT / 'docs/reviews/osic_graph_schema_transition.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'added_fields'}, indent=2))
