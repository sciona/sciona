"""Synthetic reuse invariants and an independent finite-state relaxation oracle."""
from copy import deepcopy
from itertools import product
import json
import random

import pytest

from sciona.resource_route_planning import plan
from sciona.resource_route_graph import build_resource_route_graph
from scripts.resource_route_synthetic import payload
from tests.test_text_pair_provider_graphs import run


def oracle(p):
    """Bellman-Ford over the full bounded product, without candidate filtering."""
    if p['start'] in p['blocked_nodes'] or p['goal'] in p['blocked_nodes']:
        return None
    states = list(product(p['nodes'], product(*(range(b+1) for b in p['budgets']))))
    distance = {(p['start'], (0,)*len(p['budgets'])): 0}
    for _ in range(len(states)-1):
        previous = distance.copy()
        for (node, used), cost in previous.items():
            for edge in p['edges']:
                if edge['source'] != node or edge['id'] in p['blocked_edges']:
                    continue
                if edge['target'] in p['blocked_nodes']:
                    continue
                total = tuple(a+b for a,b in zip(used, edge['resources']))
                if any(a>b for a,b in zip(total,p['budgets'])):
                    continue
                key = edge['target'],total
                candidate = cost+edge['cost']
                if key not in distance or candidate < distance[key]:
                    distance[key] = candidate
        if previous == distance:
            break
    return min((cost for (node,_),cost in distance.items() if node == p['goal']), default=None)


def check_witness(p, result):
    edges = {e['id']:e for e in p['edges']}
    node = p['start'];cost = 0;used = [0]*len(p['budgets'])
    for identity in result['edge_ids']:
        edge = edges[identity]
        assert identity not in p['blocked_edges'] and edge['source'] == node
        node = edge['target']
        assert node not in p['blocked_nodes']
        cost += edge['cost']
        used = [a+b for a,b in zip(used,edge['resources'])]
    assert node == p['goal'] and cost == result['cost'] and used == result['resources']
    assert all(a<=b for a,b in zip(used,p['budgets']))


def test_multiresource_zero_cycles_against_independent_relaxation():
    rng = random.Random(731)
    for _ in range(60):
        p = dict(nodes=['s','a','b','g'], edges=[], start='s', goal='g',
                 budgets=[2,3], blocked_nodes=[], blocked_edges=[], max_expansions=1000)
        for i in range(12):
            p['edges'].append(dict(id=str(i),source=rng.choice(p['nodes']),target=rng.choice(p['nodes']),
                                  cost=rng.randrange(4),resources=[rng.randrange(2),rng.randrange(3)]))
        p['edges'].append(dict(id='zero-loop',source='s',target='s',cost=0,resources=[0,0]))
        if rng.randrange(3)==0:p['blocked_nodes']=['a']
        if rng.randrange(2)==0:p['blocked_edges']=['0','1']
        expected = oracle(p);result = plan(p)
        assert result['status'] == ('infeasible' if expected is None else 'optimal')
        assert result['cost'] == expected
        if expected is not None:check_witness(p,result)


@pytest.mark.parametrize('change', ['rename', 'cost_scale', 'resource_scale', 'resource_permutation'])
def test_meaning_is_caller_policy(change):
    p = payload();p['budgets'].append(10)
    for edge in p['edges']:edge['resources'].append(1)
    before = plan(p);q = deepcopy(p)
    if change == 'rename':
        names = {n:'opaque-'+str(i) for i,n in enumerate(p['nodes'])}
        q['nodes'] = [names[n] for n in q['nodes']]
        q['start'],q['goal'] = names[q['start']],names[q['goal']]
        for edge in q['edges']:
            edge['source'],edge['target'] = names[edge['source']],names[edge['target']]
    elif change == 'cost_scale':
        for edge in q['edges']:edge['cost'] *= 7
    else:
        transform = (lambda v:[v[0]*5,v[1]*3]) if change == 'resource_scale' else (lambda v:v[::-1])
        q['budgets'] = transform(q['budgets'])
        for edge in q['edges']:edge['resources'] = transform(edge['resources'])
    after = plan(q)
    assert after['status'] == before['status'] == 'optimal'
    assert after['edge_ids'] == before['edge_ids']
    assert after['cost'] == before['cost']*(7 if change == 'cost_scale' else 1)
    check_witness(q,after)


@pytest.mark.parametrize('case', ['optimal', 'infeasible', 'search_limit', 'identity'])
def test_native_stages_preserve_state_and_json(case,tmp_path,monkeypatch):
    p = payload()
    if case == 'infeasible':p['blocked_edges']=['detour']
    if case == 'search_limit':p['max_expansions']=1
    if case == 'identity':p['goal']=p['start']
    original = deepcopy(p);graph = build_resource_route_graph()
    captured = run(graph,{'payload':p},tmp_path,monkeypatch)
    for edge in graph.edges:
        assert captured[edge.source_id,'out_'+edge.output_name] is captured[edge.target_id,'in_'+edge.input_name]
    result = captured['select','out_result']
    assert result == plan(p)
    assert result['status'] == ('optimal' if case == 'identity' else case)
    assert json.loads(json.dumps(result,allow_nan=False)) == result
    assert p == original
