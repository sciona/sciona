"""Check exact DSB staged identities and interfaces without claiming approval."""
import json
from scripts.inventory_dsb_decomposed_providers import inventory, ROOT
from scripts.plan_dsb_provider_interfaces import plan as interface_plan
from scripts.stage_dsb_provider_dependencies import expected, current_rows

PLAN=ROOT/'docs/reviews/dsb_dependency_intake_plan.json'


def contracts():
    result=interface_plan()
    if result!=json.loads((ROOT/'docs/reviews/dsb_provider_interface_plan.json').read_text()):
        raise ValueError('Interface evidence changed')
    return {r['runtime_fqdn']:r for r in result['provider_contracts']}


def qualify_plan():
    expected()
    current=inventory()
    return dict(records=current['providers'],interfaces=contracts())


def check(db,plan):
    declarations,_=expected()
    versions=sorted({r['dependent_version_id'] for r in declarations})
    key=lambda r:tuple(r[k] for k in ('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash','port_name'))
    if current_rows(db,versions)!=sorted(declarations,key=key):raise ValueError('Dependency declarations changed')
    for record in plan['records']:
        for prefix,table,key in [('atom','atoms','atom_id'),('artifact','artifacts','artifact_id')]:
            rows=db.execute(f'''SELECT a.{key}::text AS identity,a.status,a.is_publishable,
                v.version_id::text,v.content_hash,v.trust_tier,{prefix}_is_publishable(a.{key}) AS native_eligible
                FROM {table} a JOIN {prefix}_versions v USING({key}) WHERE a.fqdn=%s AND v.is_latest''',
                (record['catalog_fqdn'],)).fetchall()
            if rows!=record[prefix+'_latest']:raise ValueError('Staged provider state changed')
            if len(rows)!=1 or rows[0]['version_id']!=record['version_id'] or rows[0]['content_hash']!=record['content_hash']:
                raise ValueError('Exact provider version missing')
            fields=('direction','name','ordinal','type_desc','required','default_value_repr')
            ports=db.execute('SELECT '+','.join(fields)+f' FROM {prefix}_io_specs WHERE version_id=%s',
                             (record['version_id'],)).fetchall()
            wanted=[{k:p[k] for k in fields} for p in plan['interfaces'][record['runtime_fqdn']]['ports']]
            # Existing graph-object ports may have more concrete Python annotations;
            # the interface plan retains the reviewed graph boundary types.
            order=lambda p:(p['direction'],p['ordinal'])
            if sorted(ports,key=order)!=sorted(wanted,key=order):raise ValueError('Staged provider ports changed')
