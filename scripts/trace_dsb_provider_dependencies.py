"""Trace registered provider calls during the independent DSB graph qualification.

Only provider identities and aggregate call counts are recorded. No arguments,
return values, local variables, model state or input provenance are captured.
"""
import argparse
import asyncio
from collections import Counter
import importlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

from sciona.ghost.registry import REGISTRY
from scripts.inventory_dsb_decomposed_providers import inventory, ROOT, sha
from scripts import validate_dsb_lifecycle_contract_graph as qualification


def code_index():
    result = {}
    for name, entry in REGISTRY.items():
        function = inspect.unwrap(entry['impl'])
        if not inspect.isfunction(function):
            continue
        code = function.__code__
        if code in result and result[code] != name:
            raise ValueError('Ambiguous registered function code')
        result[code] = name
    return result


async def trace(source_root):
    current = inventory(offline=True)
    expected = {r['runtime_fqdn'] for r in current['providers']}
    for name in expected:
        importlib.import_module(name.rsplit('.', 1)[0])
    codes = code_index()
    original_execute = qualification.oracle.runner.CDGExecutionSession.execute
    executions = []

    async def execute(session, *args, **kwargs):
        if sys.getprofile() is not None:
            raise ValueError('Existing profiler must not be overwritten')
        counts, edges, roots = Counter(), Counter(), Counter()
        def profile(frame, event, arg):
            if event != 'call':
                return
            target = codes.get(frame.f_code)
            if target is None:
                return
            counts[target] += 1
            parent = frame.f_back
            while parent is not None and parent.f_code not in codes:
                parent = parent.f_back
            if parent is None:
                roots[target] += 1
            else:
                edges[(codes[parent.f_code], target)] += 1
        try:
            sys.setprofile(profile)
            result = await original_execute(session, *args, **kwargs)
        finally:
            sys.setprofile(None)
        if result['status'] != 'completed':
            raise ValueError('Complete graph execution required')
        unexpected = set(counts) - expected
        if unexpected:
            raise ValueError('Provider inventory omitted observed calls: ' + str(sorted(unexpected)))
        executions.append(dict(calls=dict(sorted(counts.items())), root_calls=dict(sorted(roots.items())),
            dependencies=[dict(caller=a, callee=b, calls=n) for (a,b),n in sorted(edges.items())]))
        return result

    with patch.object(qualification.oracle.runner.CDGExecutionSession, 'execute', execute):
        evidence = await qualification.validate(source_root)
    observed = {name for run in executions for name in run['calls']}
    edges = {(edge['caller'],edge['callee']) for run in executions for edge in run['dependencies']}
    if len(executions) != 2 or observed != expected:
        raise ValueError('Every inventoried provider must be exercised across both qualification cases')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        graph_digest=current['graph_digest'], providers=len(expected), observed_dependency_links=len(edges),
        complete_static_closure=False, executions=executions, cases=evidence['cases'],
        dependencies=[dict(caller=a,callee=b) for a,b in sorted(edges)],
        inventory_sha256=sha(ROOT/'docs/reviews/dsb_decomposed_provider_inventory.json'),
        validator_sha256=sha(Path(__file__)), qualification_sha256=sha(Path(qualification.__file__)),
        limitations=['Runtime evidence covers the two qualified synthetic three-epoch windows only.',
            'Dependencies are attributed to the nearest registered caller during execution.',
            'Returned models and factories execute under their consuming provider; static construction/deferred-use edges require separate review.',
            'Unexecuted branches and arbitrary caller callbacks are not certified by this trace.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    result=asyncio.run(trace(parser.parse_args().source_root))
    (ROOT/'docs/reviews/dsb_provider_runtime_dependencies.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','providers','observed_dependency_links','complete_static_closure')}))
