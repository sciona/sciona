"""Execute the qualified population graph with six exact served provider bindings."""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from scripts import validate_osic_image_population_graph as qualification
from scripts.import_flavours_provider_drafts import connect

ROOT=qualification.overlay.ROOT
REVISION='ba481f9b'
REVIEWS={
    'osic_image_semantics':'d038c287',
    'osic_image_lifecycle_semantics':'239507e1',
    'keras_bounded_semantics':'5715edee',
    'ordered_fields_semantics':'39bf03c1',
    'osic_image_routing_semantics':'a5fead3e',
}


def immutable(name,revision):
    data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    if data!=(ROOT/name).read_bytes():raise ValueError('Immutable graph qualification changed')
    return json.loads(data)


def worker(state_directory,dependencies):
    from sciona.ghost.registry import REGISTRY
    records={}
    for name,revision in REVIEWS.items():
        review=immutable('docs/reviews/'+name+'.json',revision)
        for record in review['providers']:
            records[record['fqdn']]=(record,review['software_sha256']['provider'])
    decoded=qualification.decode_execution_graph
    bindings=[]
    def resolve(nodes,edges,digest):
        graph=decoded(nodes,edges,digest)
        # Every projected edge is authoritative, including callable injections.
        for index in range(len(edges)):
            try:decoded(nodes,edges[:index]+edges[index+1:],digest)
            except ValueError:pass
            else:raise ValueError('Missing projected edge accepted')
        with connect() as db:
            db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            for node in graph.nodes:
                record,provider_hash=records[node.matched_primitive]
                module,symbol=record['fqdn'].rsplit('.',1)
                imported=importlib.import_module(module)
                if qualification.overlay.sha(inspect.getfile(imported))!=provider_hash:
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
        execution=qualification.worker(state_directory,dependencies)
    baseline=immutable('docs/reviews/osic_image_population_graph.json',REVISION)['execution']
    if execution!=baseline or len(bindings)!=6:
        raise ValueError('Served graph did not exactly reproduce qualified execution')
    return dict(passed=True,execution=execution,bindings=bindings,served_catalogs=2,
                missing_edge_projections_rejected=10,original_workflow_approved=False)


def validate(dependencies,source_directory,state_directory):
    baseline=immutable('docs/reviews/osic_image_population_graph.json',REVISION)
    paths=list(baseline['software_sha256'])+['scripts/verify_osic_population_graph_served.py']
    paths += ['docs/reviews/'+name+'.json' for name in REVIEWS]
    before={name:qualification.overlay.sha(ROOT/name) for name in paths}
    for name,digest in baseline['software_sha256'].items():
        if before[name]!=digest:raise ValueError('Qualified graph implementation changed')
    run=subprocess.run
    redirects=0
    def redirect(command,*args,**kwargs):
        nonlocal redirects
        command=list(command)
        target=str(Path(qualification.overlay.__file__).resolve())
        if target in command and '--worker' in command:
            command[command.index(target)]=str(Path(__file__).resolve());redirects+=1
        return run(command,*args,**kwargs)
    with patch.object(qualification.overlay.subprocess,'run',redirect):
        result=qualification.overlay.validate(dependencies,source_directory,state_directory)
    if redirects!=1 or before!={name:qualification.overlay.sha(ROOT/name) for name in paths}:
        raise ValueError('Served graph inputs or worker dispatch changed')
    if result['execution']['execution']!=baseline['execution']:
        raise ValueError('Served worker differs from immutable graph evidence')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        served_graph=result['execution'],software_sha256=before,
        dependency_archive_sha256=result['dependency_archive_sha256'],shared_environment_mutations=0,
        limitations=[
            'Six Tier 3 served providers execute through the canonical codec and real sequential graph executor; the graph itself and original CDG remain unapproved.',
            'Synthetic two-entity four-query graph wiring over intermediate numeric state, compared to qualified direct provider composition. No independent source-notebook replay or empirical effectiveness claim.',
            'Explicit discovery and intercepted intermediate persistence; HTTP transport, default discovery and caching remain unqualified.',
            'Complete training, quantile graph integration and final branch blending remain outstanding.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    for name in ('dependencies','source-directory','state-directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.worker:print(json.dumps(worker(args.state_directory,args.dependencies)))
    else:
        result=validate(args.dependencies,args.source_directory,args.state_directory)
        (ROOT/'docs/reviews/osic_population_graph_served.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,nodes=6,served_catalogs=2,exact_predictions=True,approved=False)))
