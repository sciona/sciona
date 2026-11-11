"""Review all changed callable identities against immutable relocation sources."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from dotenv import dotenv_values
import psycopg
from psycopg.rows import dict_row

from scripts.plan_osic_relocated_provider_versions import ROOT, material, normalized, plan

ANCHOR='c5d9098ee9c6f3e1dc6da34e5ffd2a31efd434d5'
PREVIOUS='bfe6e9bd582bc67d77ad7c5bb779b797df420b8d'


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def review():
    prior=plan()
    repo=ROOT.parent/'sciona-atoms-ml'
    anchor_report=json.loads(subprocess.check_output(['git','show',
        ANCHOR+':docs/reviews/osic_runtime_source_anchor.json'],cwd=repo))
    def source_check():
        for name,digest in anchor_report['qualified_source_sha256'].items():
            raw=subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=repo)
            if hashlib.sha256(raw).hexdigest()!=digest or raw!=(repo/name).read_bytes():
                raise ValueError('Committed qualified provider source changed')
    source_check()
    runtime_count=0
    for name in anchor_report['qualified_source_sha256']:
        if name.startswith('src/sciona/_runtime/ml/') and not name.endswith('/__init__.py'):
            old=subprocess.check_output(['git','show','5ed15bda:sciona/'+Path(name).name],cwd=ROOT)
            if normalized(old)!=normalized((repo/name).read_bytes()):
                raise ValueError('Transitive implementation changes exceed import relocation')
            runtime_count+=1
    evidence={}
    for revision,name in [('f28f2dde','osic_relocated_combined_graph'),
                          ('70ea1ddc','osic_relocated_provider_contracts')]:
        path='docs/reviews/'+name+'.json'
        raw=subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT)
        if raw!=(ROOT/path).read_bytes() or json.loads(raw)['passed'] is not True:
            raise ValueError('Immutable relocation evidence differs')
        evidence[name]=dict(revision=revision,sha256=hashlib.sha256(raw).hexdigest())
    modules=sorted({r['runtime_fqdn'].rsplit('.',1)[0] for r in prior['records']
                    if r['action']=='new_provider_version_required'})
    graph_targets={r['runtime_fqdn'] for r in prior['records']}
    records=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
            row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        for module in modules:
            path=repo/'src'/Path(*module.split('.')).with_suffix('.py')
            source=path.read_text()
            old_source=subprocess.check_output(['git','show',PREVIOUS+':'+str(path.relative_to(repo))],cwd=repo).decode()
            if normalized(source)!=normalized(old_source):
                raise ValueError('Provider changes exceed import relocation')
            for node in ast.parse(source).body:
                if not isinstance(node,ast.FunctionDef) or not any(isinstance(d,ast.Call)
                        and isinstance(d.func,ast.Name) and d.func.id=='register_atom' for d in node.decorator_list):
                    continue
                target=module+'.'+node.name
                rows=db.execute('''SELECT a.atom_id::text,a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM atoms a JOIN atom_versions v USING(atom_id)
                    WHERE a.import_module=%s AND a.source_symbol=%s AND v.is_latest''',(module,node.name)).fetchall()
                if len(rows)!=1:
                    raise ValueError('Unique current provider required')
                old=rows[0]
                old_material=material(repo,path,old['fqdn'],node.name,old_source)
                proposed=material(repo,path,old['fqdn'],node.name,source)
                if old_material['version_id']!=old['version_id'] or old_material['content_hash']!=old['content_hash']:
                    raise ValueError('Prior source and approved version differ')
                if proposed['content_hash']==old['content_hash'] or old['trust_tier']!=3:
                    raise ValueError('Expected a changed Tier 3 callable')
                contracts={}
                for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                    version=db.execute(f'''SELECT version_id::text,content_hash,trust_tier FROM {prefix}_versions
                        WHERE {key}=%s AND is_latest''',(old['atom_id'],)).fetchall()
                    if version!=[{k:old[k] for k in ('version_id','content_hash','trust_tier')}]:
                        raise ValueError('Catalog mirror differs')
                    if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(old['atom_id'],)).fetchone():
                        raise ValueError('Prior provider is not served')
                    fields='direction,ordinal,name,type_desc,required,default_value_repr,constraints'
                    contracts[prefix]=db.execute(f'SELECT {fields} FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                                                 (old['version_id'],)).fetchall()
                if not contracts['atom'] or contracts['atom']!=contracts['artifact']:
                    raise ValueError('Versioned contracts differ')
                if [p['name'] for p in contracts['atom'] if p['direction']=='input']!=[a.arg for a in node.args.args]:
                    raise ValueError('Input signature and contract differ')
                failed=db.execute("SELECT count(*) AS n FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed AND status IN ('completed','failed')",
                                  (old['version_id'],)).fetchone()['n']
                if failed:
                    raise ValueError('Prior version has unresolved failed audit')
                dependencies=db.execute('''SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,
                    port_name,optional,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s
                    ORDER BY dependency_artifact_fqdn,port_name''',(old['version_id'],)).fetchall()
                for dependency in dependencies:
                    found=db.execute('''SELECT v.version_id FROM artifacts a JOIN artifact_versions v USING(artifact_id)
                        WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest AND a.status='approved' AND a.is_publishable''',
                        (dependency['dependency_artifact_fqdn'],dependency['dependency_content_hash'])).fetchall()
                    if len(found)!=1:
                        raise ValueError('Registered dependency is not exact and approved')
                records.append(dict(runtime_fqdn=target,artifact_id=old['atom_id'],
                    previous_version_id=old['version_id'],previous_content_hash=old['content_hash'],proposed=proposed,
                    provider_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    interface_contract_sha256=canonical_digest(contracts['atom']),interface_ports=len(contracts['atom']),
                    dependencies_sha256=canonical_digest(dependencies),registered_dependencies=len(dependencies),
                    used_by_combined_graph=target in graph_targets,
                    wrapper_import_only=True,prior_approval_and_contracts_verified=True))
    source_check()
    if len(records)!=15 or sum(r['used_by_combined_graph'] for r in records)!=9 or runtime_count!=22:
        raise ValueError('Unexpected relocation review scope')
    return dict(format='osic-relocated-version-review.v1',passed=True,approved=False,catalog_mutations=0,
        proposed_tier=3,verdict='import_relocation_equivalence_verified',provider_revision=ANCHOR,
        previous_provider_revision=PREVIOUS,transitive_runtime_modules_compared=runtime_count,
        providers=records,evidence=evidence,reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remaining_publication_gates=[
            'Review unchanged source-specific validity limits and generic numerical dependency identities in the publication payload.',
            'Expand transaction and fault-injection gates from nine graph callables to all fifteen changed callables.',
            'Publish versions atomically, preserve historical versions and verify both served catalogs and exact runtime identities.',
            'Qualify replacement graph versions with the new bindings; complete original image training and workflow separately.'],
        limitations=[
            'Automated Tier 3 source/import equivalence review, not human Tier 1 or empirical Tier 2 validation.',
            'Fresh combined numerical execution directly exercises nine changed callables; the six additional wrappers are covered here by source equivalence, transitive runtime comparison and unchanged prior contracts.',
            'No API, numeric policy, state format, source recipe or reuse boundary is generalized by relocation.',
            'Competition-specific preparation and policies remain adapters; generic affine fitting, weighted summation and loss primitives remain separate.',
            'No catalog updates, release merge, clean-installation claim or original-CDG approval.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/osic_relocated_version_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),runtime_modules=result['transitive_runtime_modules_compared'],approved=False)))
