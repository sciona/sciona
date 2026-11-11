"""Freeze candidate ports for exact missing lifecycle dependencies; no writes to DB."""
import hashlib
import inspect
import json
from pathlib import Path

from sciona.lifecycle_provider_contracts import contracts
from scripts.inventory_nasa_first_workflow_providers import inventory
from sciona.ghost.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
FORMATTER = 'sciona.atoms.ml.domain_adapters.first_place_prediction.format_predictions'


def compatible(runtime, side, local, canonical):
    if local['type_desc'] == canonical['type_desc']:
        return True
    return (runtime == FORMATTER and side == 'inputs' and local['name'] == 'values'
            and local['type_desc'] == 'NDArray[np.int64]' and canonical['type_desc'] == 'np.ndarray')


def plan():
    current = inventory()
    records = []
    for provider in current['providers']:
        if provider['served']:
            continue
        runtime = provider['runtime_fqdn']
        if provider['role'] == 'reusable_operation':
            inputs, outputs = contracts(runtime.split('.model_selection.')[1])
        else:
            # Domain adapters are intentionally source-specific. Retain every
            # usage constraint; do not silently widen conflicting port types.
            sides = []
            for side in ['inputs', 'outputs']:
                variants = [site[side] for site in provider['graph_uses']]
                names = [[port['name'] for port in ports] for ports in variants]
                if any(value != names[0] for value in names):
                    raise ValueError('Domain port ordering differs: ' + runtime)
                ports = []
                for index, name in enumerate(names[0]):
                    alternatives = [ports[index] for ports in variants]
                    types = {p['type_desc'] for p in alternatives}
                    formatter_values = runtime == FORMATTER and side == 'inputs' and name == 'values'
                    if len(types) != 1 and not (formatter_values and types == {'np.ndarray','NDArray[np.int64]'}):
                        raise ValueError('Domain port type differs: ' + runtime + '/' + name)
                    port = dict(alternatives[0])
                    port['constraints'] = ' '.join(dict.fromkeys(p['constraints'] for p in alternatives))
                    if formatter_values:
                        port['type_desc'] = 'np.ndarray'
                        port['constraints'] = 'Positional query-aligned prediction vector in minutes; preserve int64 primary values and fractional float64 fallback values without casting. Empty output is float64.'
                    if not port['constraints']:
                        raise ValueError('Missing domain contract: ' + runtime + '/' + name)
                    ports.append(port)
                sides.append(ports)
            inputs, outputs = sides
        if [p['name'] for p in inputs] != list(inspect.signature(REGISTRY[runtime]['impl']).parameters):
            raise ValueError('Callable input order differs: ' + runtime)
        for site in provider['graph_uses']:
            for side, canonical in [('inputs', inputs), ('outputs', outputs)]:
                if side not in site:
                    continue
                if [p['name'] for p in site[side]] != [p['name'] for p in canonical]:
                    raise ValueError('Graph port order differs: ' + runtime)
                if not all(compatible(runtime, side, local, target) for local,target in zip(site[side],canonical)):
                    raise ValueError('Graph type differs: ' + runtime)
        records.append(dict(runtime_fqdn=runtime, fqdn=provider['fqdn'], role=provider['role'],
            version_id=provider['version_id'], content_hash=provider['content_hash'],
            provider_source=provider['provider_source'], provider_sha256=provider['provider_sha256'],
            inputs=inputs, outputs=outputs))
    return dict(passed=True, approved=False, catalog_mutations=0, candidate_tier=3,
        providers=records, provider_count=len(records),
        reusable_operations=sum(p['role']=='reusable_operation' for p in records),
        domain_adapters=sum(p['role']=='domain_adapter' for p in records),
        graphs=current['graph_records'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'sciona/lifecycle_provider_contracts.py', Path(__file__), ROOT/'scripts/inventory_nasa_first_workflow_providers.py']},
        limitations=['Candidate semantic contracts, not catalog staging or approval.',
                     'Domain adapter ports retain source-specific constraints from the qualified graphs.',
                     'Runtime/license evidence, cross-domain execution and database gates remain required.'])


if __name__ == '__main__':
    report = plan()
    (ROOT/'docs/reviews/competition_nasa_first_lifecycle_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','provider_count','reusable_operations','domain_adapters','catalog_mutations']}))
