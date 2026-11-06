"""Freeze source-matched catalog ports for nine approved numerics and one draft recipe."""
import json
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_video_multilabel_graph import build_decomposed_video_multilabel_graph
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.stage_video_sharded_recipe import reviewed_plan, sha
from scripts.promote_video_numeric_providers import qualify_plan, check

PATH=ROOT/'docs/reviews/video_graph_contracts.json'
PORTS='direction,name,ordinal,type_desc,required,default_value_repr,constraints'


def freeze():
    recipe=reviewed_plan()['providers'][0]
    publication=qualify_plan()
    graph=build_decomposed_video_multilabel_graph()
    wanted={n.matched_primitive for n in graph.nodes}
    directory=ROOT.parent/'sciona-atoms-ml'
    specs={s.import_module+'.'+s.source_symbol:s for s in _parse_registered_atoms(
        repo=ProviderRepo(directory.name,directory),artifact_root=directory/'src/sciona/atoms')
        if s.import_module+'.'+s.source_symbol in wanted}
    if set(specs)!=wanted:raise ValueError('Complete source provider inventory required')
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check(db,publication,published=True)
        for node in graph.nodes:
            spec=specs[node.matched_primitive]
            rows=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
                v.version_id::text,v.content_hash,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id)
                WHERE a.fqdn=%s AND v.is_latest''',(spec.fqdn,)).fetchall()
            if len(rows)!=1:raise ValueError('Unique latest provider required')
            row=rows[0];draft=node.node_id=='execute'
            if (row['version_id']!=str(spec.version_id) or row['content_hash']!=spec.content_hash
                    or row['trust_tier']!=3 or row['status']!=('draft' if draft else 'approved')
                    or row['is_publishable']!= (not draft)):
                raise ValueError('Source/version/state differs: '+node.node_id)
            if draft and row['artifact_id']!=recipe['atom_id']:raise ValueError('Exact staged recipe required')
            record=dict(node_id=node.node_id,runtime_fqdn=node.matched_primitive,**row)
            for prefix,key,field in [('artifact','artifact_id','ports'),('atom','atom_id','legacy_ports')]:
                record[field]=db.execute(f'SELECT {PORTS} FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                    (row['version_id'],)).fetchall()
                if not draft:
                    if not db.execute(f'SELECT {prefix}_is_publishable(%s) AS eligible',(row['artifact_id'],)).fetchone()['eligible']:
                        raise ValueError('Numeric provider not eligible')
                    if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(row['artifact_id'],)).fetchone():
                        raise ValueError('Numeric provider not served')
            if not record['ports']:raise ValueError('Missing provider ports')
            records.append(record)
    return dict(format='video-graph-contracts.v1',read_only=True,catalog_mutations=0,approved=False,
        records=records,checker_sha256=sha(__file__),
        limitations=['Recipe is staged, not approved. These ports qualify a corrected graph execution, not publication.',
                    'Exact source versions and numeric eligibility checked; original CDG promotion remains separate.'])


if __name__=='__main__':
    result=freeze();PATH.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(approved_numerics=9,draft_recipes=1,catalog_mutations=0)))
