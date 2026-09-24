"""Source, contract and catalog identity preflight for adaptive history."""
import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import sciona.atoms.ml.calibration.adaptive_history as provider
from sciona.adaptive_history_graph import build_adaptive_history_graph
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_conditional_correction import ROOT, SOURCE_VERSION, SOURCE_HASH, require, sha
from scripts.review_conditional_correction_environment import audit as environment_audit

FQDN = 'cdg.ml.statistics.adaptive_history'


def review():
    directory = ROOT / 'docs/reviews'
    source = json.loads((directory/'adaptive_history_source_comparison.json').read_text())
    execution = json.loads((directory/'adaptive_history_graph_execution.json').read_text())
    require(source['passed'] is True and source['source_cases']==128 and source['empty_windows']>0, 'Source comparisons missing')
    require(source['source_sha256']=='cbcbe972af0d63a240923570e840744d573d972ee0bce19e04bf55a2c8b01444', 'Source pin differs')
    require(source['validator_sha256']==sha(ROOT/'scripts/validate_adaptive_history_source.py'), 'Source validator drift')
    require(source['provider_sha256']==execution['provider_sha256']==sha(Path(provider.__file__)), 'Provider evidence drift')
    for name,digest in execution['implementation_sha256'].items():
        require(name in {'sciona/adaptive_history_graph.py','scripts/validate_adaptive_history_graph.py','tests/test_adaptive_history.py'}, 'Unexpected evidence file')
        require(sha(ROOT/name)==digest,'Execution evidence drift')
    require(len(execution['implementation_sha256'])==3,'Missing implementation evidence')
    require(execution['passed'] is True and execution['dimensional_witness_passed'] is True,'Graph execution missing')
    require(execution['scenarios']==[dict(domain=domain,empty_window=empty,completed=True,output_ports=3)
        for domain in ['manufacturing_measurements','energy_sensor_readings'] for empty in [False,True]],'Required graph scenarios missing')
    graph=build_adaptive_history_graph();digest,_,_=encode_execution_graph(graph)
    require(execution['graph_sha256']==digest,'Graph digest differs')
    node=graph.nodes[0]
    require(list(inspect.signature(provider.adaptive_history_statistics).parameters)==[p.name for p in node.inputs], 'Callable ports differ')
    repo=ROOT.parent/'sciona-atoms-ml'
    specs=_parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',repo),artifact_root=repo/'src/sciona/atoms')
    matches=[s for s in specs if s.import_module+'.'+s.source_symbol==node.matched_primitive]
    require(len(matches)==1,'Unique provider inventory required');spec=matches[0]
    require(spec.file_path.resolve()==Path(provider.__file__).resolve(),'Provider import shadowed')
    atom_id=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)
    graph_id=uuid5(NAMESPACE_URL,'sciona-reusable-cdg:'+FQDN)
    environment=environment_audit()
    require(environment['provisioned_direct_dependencies_compatible'],'Provider dependencies differ')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        original=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v '
            'USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        require(original==dict(status='draft',is_publishable=False,content_hash=SOURCE_HASH),'Original intake differs')
        identities=[]
        for fqdn,identity in [(FQDN,graph_id),(spec.fqdn,atom_id)]:
            rows=db.execute('SELECT artifact_id,status,is_publishable FROM artifacts WHERE fqdn=%s',(fqdn,)).fetchall()
            require(len(rows)<=1 and all(str(r['artifact_id'])==str(identity) for r in rows),'Catalog identity conflict')
            identities.append(dict(fqdn=fqdn,artifact_id=str(identity),exists=bool(rows)))
    return dict(format='adaptive-history-review.v1',semantic_verdict='acceptable_with_limits',proposed_tier=3,
        approved=False,catalog_mutations=0,graph_sha256=digest,graph_version_id=str(uuid5(graph_id,digest)),
        provider_version_id=str(spec.version_id),provider_content_hash=spec.content_hash,
        provider_sha256=source['provider_sha256'],identities=identities,input_ports=5,output_ports=3,
        evidence_sha256={name:sha(directory/name) for name in ['adaptive_history_source_comparison.json','adaptive_history_graph_execution.json']},
        license_notice_sha256=environment['license_sha256'],reviewer_sha256=sha(Path(__file__)),
        limitations=source['limitations']+execution['limitations'],
        remaining=['Transactional catalog import and exact served-version qualification.',
                   'Negative publication gates, rollback/apply/idempotency, and served execution.',
                   'Complete original feature pipeline, source null handling, and domain adapters remain separate.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/adaptive_history_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['semantic_verdict','input_ports','output_ports','identities','approved']}))
