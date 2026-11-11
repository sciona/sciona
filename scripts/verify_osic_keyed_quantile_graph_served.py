"""Execute the qualified quantile replay graph with seven exact served provider bindings."""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from scripts import validate_osic_keyed_quantile_graph as qualification
from scripts.import_flavours_provider_drafts import connect

ROOT=qualification.reference.ROOT
REVISION='78124515'
REVIEWS={
    'osic_query_routing_semantics':'c92d4a1f',
    'osic_feature_replay_semantics':'583e7272',
    'osic_feature_projection_semantics':'ea47ddc9',
    'typed_inputs_semantics':'334a7878',
    'osic_quantile_semantics':'cfbf9be2',
}


def immutable(name,revision):
    data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    if data!=(ROOT/name).read_bytes():raise ValueError('Immutable graph qualification changed')
    return json.loads(data)


def child(directory):
    from sciona.ghost.registry import REGISTRY
    baseline=immutable('docs/reviews/osic_keyed_quantile_graph.json',REVISION)
    records={}
    for name,revision in REVIEWS.items():
        review=immutable('docs/reviews/'+name+'.json',revision)
        for record in review['providers']:
            provider_hash=review['software_sha256']['provider']
            if provider_hash != baseline['graph_execution']['provider_sha256'][record['fqdn'].rsplit('.',1)[0]]:
                raise ValueError('Graph implementation differs from semantic review')
            records[record['fqdn']]=(record,provider_hash)
    decoded=qualification.decode_execution_graph
    bindings=[]
    def resolve(nodes,edges,digest):
        graph=decoded(nodes,edges,digest)
        # The qualification also sends each missing-edge projection here;
        # decoding must reject it before any catalog binding is accepted.
        with connect() as db:
            db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            for node in graph.nodes:
                record,provider_hash=records[node.matched_primitive]
                module,symbol=record['fqdn'].rsplit('.',1)
                imported=importlib.import_module(module)
                if qualification.reference.sha(Path(inspect.getfile(imported)))!=provider_hash:
                    raise ValueError('Approved implementation changed')
                expected=dict(fqdn=record['fqdn'],version_id=record['version_id'],content_hash=record['content_hash'],
                              trust_tier=3,import_module=module,source_symbol=symbol)
                for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                    rows=db.execute(f'''SELECT s.fqdn,v.version_id::text,v.content_hash,v.trust_tier,
                        a.import_module,a.source_symbol FROM catalog_{prefix}s_served s
                        JOIN {prefix}_versions v USING({key}) JOIN atoms a ON a.atom_id=s.{key}
                        WHERE s.{key}=%s AND v.is_latest''',(record['artifact_id'],)).fetchall()
                    if not rows or any(row!=expected for row in rows):
                        raise ValueError('Exact served binding required in both catalogs')
                fn=getattr(imported,symbol)
                if REGISTRY[record['fqdn']]['impl'] is not fn:
                    raise ValueError('Registry differs from served implementation')
                if list(inspect.signature(fn).parameters)!=[port.name for port in node.inputs]:
                    raise ValueError('Graph callable signature differs')
                bindings.append(dict(node_id=node.node_id,**expected))
            original=db.execute('SELECT status,is_publishable FROM artifacts WHERE fqdn=%s',
                ('cdg.competition.solution.kaggle.osic_pulmonary_1st',)).fetchone()
            if original!=dict(status='draft',is_publishable=False):
                raise ValueError('Original workflow scope changed')
        return graph
    with patch.object(qualification,'decode_execution_graph',resolve):
        execution=qualification.child(directory)
    graph = execution.pop('graph_execution')
    if graph != baseline['graph_execution'] or execution != baseline['raw_replay_qualification']['fresh_process_replay'] or len(bindings) != 7:
        raise ValueError('Served graph did not exactly reproduce qualified execution')
    return dict(execution, served_graph=dict(bindings=bindings, graph_execution=graph,
        served_catalogs=2, original_workflow_approved=False))


def validate():
    baseline=immutable('docs/reviews/osic_keyed_quantile_graph.json',REVISION)
    paths=list(baseline['software_sha256'])+['scripts/verify_osic_keyed_quantile_graph_served.py']
    paths += ['docs/reviews/'+name+'.json' for name in REVIEWS]
    before={name:qualification.reference.sha(ROOT/name) for name in paths}
    for name,digest in baseline['software_sha256'].items():
        if before[name]!=digest:raise ValueError('Qualified graph implementation changed')
    run=subprocess.run
    redirects=0
    def redirect(command,*args,**kwargs):
        nonlocal redirects
        command=list(command)
        target=str(Path(qualification.reference.__file__).resolve())
        if target in command and '--replay-directory' in command:
            command[command.index(target)]=str(Path(__file__).resolve());redirects+=1
        return run(command,*args,**kwargs)
    with patch.object(qualification.reference.subprocess,'run',redirect):
        result=qualification.reference.validate()
    served=result['fresh_process_replay'].pop('served_graph')
    if redirects!=1 or before!={name:qualification.reference.sha(ROOT/name) for name in paths}:
        raise ValueError('Served graph inputs or child dispatch changed')
    if result!=baseline['raw_replay_qualification']:
        raise ValueError('Served execution differs from immutable raw replay evidence')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        served_graph=served,raw_replay_qualification=result,software_sha256=before,
        limitations=[
            'Seven Tier 3 served providers execute through the canonical codec and real sequential graph executor; the graph itself and original CDG remain unapproved.',
            'Synthetic queries and two freshly trained 600-epoch fold states reproduce qualified raw replay exactly. No empirical effectiveness claim.',
            'Explicit discovery and intercepted intermediate persistence; HTTP transport, default discovery and caching remain unqualified.',
            'Complete image training and final branch blending remain outstanding.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-directory',type=Path)
    args=parser.parse_args()
    if args.replay_directory:print(json.dumps(child(args.replay_directory)))
    else:
        result=validate()
        (ROOT/'docs/reviews/osic_keyed_quantile_graph_served.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,nodes=7,served_catalogs=2,exact_predictions=True,approved=False)))
