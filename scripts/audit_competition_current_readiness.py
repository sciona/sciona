#!/usr/bin/env python3
"""Read-only current competition coverage; never approves intake templates."""
import hashlib,json
from pathlib import Path
from collections import Counter
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.competition_import import _canonical
from sciona.competition_promotion import assess_keyword_call_contract
from sciona.ghost.registry import REGISTRY
from sciona.visualizer.runner import _ensure_atoms_imported
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg


def selected_catalog_state(db, intake):
 """Separate current version evidence from historical template keyword coverage."""
 versions=db.execute('SELECT version_id,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',
  (intake['artifact_id'],)).fetchall()
 served=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s',
  (intake['artifact_id'],)).fetchone()['n']==1
 result=dict(unique_latest_version=len(versions)==1,served=served,corrected_version=False,
  execution_envelope_valid=False,active_bindings_complete=False,approved_provider_versions_exact=False,
  passed_version_audit=False,failed_version_audits=0)
 if len(versions)!=1:return result
 version=versions[0]
 result.update(version_id=str(version['version_id']),trust_tier=version['trust_tier'],
  corrected_version=str(version['version_id'])!=str(intake['version_id']))
 evidence=db.execute('SELECT passed FROM artifact_audit_evidence WHERE version_id=%s',(version['version_id'],)).fetchall()
 result['passed_version_audit']=any(row['passed'] for row in evidence)
 result['failed_version_audits']=sum(not row['passed'] for row in evidence)
 if not served:return result
 document=db.execute('SELECT get_artifact_document(%s) AS d',(intake['fqdn'],)).fetchone()['d']
 try:
  graph=_artifact_document_to_cdg(document,version_id=str(version['version_id']),content_hash=version['content_hash'],require_execution_envelope=True)
 except ValueError:
  return result
 result['execution_envelope_valid']=True
 bindings=db.execute('SELECT node_id,status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',
  (version['version_id'],)).fetchall()
 result['active_bindings_complete']=bool(graph.nodes) and len(bindings)==len(graph.nodes) and {b['node_id'] for b in bindings}=={n.node_id for n in graph.nodes} and all(b['status']=='active' for b in bindings)
 identities={(b['bound_artifact_fqdn'],b['bound_version_content_hash']) for b in bindings}
 exact=bool(identities)
 for fqdn,digest in identities:
  providers=db.execute("SELECT v.version_id FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.content_hash=%s AND a.status='approved' AND a.is_publishable AND v.is_latest",(fqdn,digest)).fetchall()
  exact=exact and len(providers)==1
 result['approved_provider_versions_exact']=exact
 result['nodes']=len(graph.nodes)
 result['distinct_bound_providers']=len(identities)
 return result


_ensure_atoms_imported()
root=Path(__file__).resolve().parents[1]
with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,options='-c default_transaction_read_only=on') as db:
 rows=db.execute("SELECT e.artifact_id,e.version_id,e.details,v.content_hash,a.fqdn,a.status,a.is_publishable FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) JOIN artifacts a ON a.artifact_id=e.artifact_id WHERE e.runner_version='competition-intake.v1' ORDER BY e.version_id").fetchall()
 catalog={}
 for r in db.execute('SELECT a.fqdn,a.import_module,a.source_symbol,a.status,a.is_publishable,v.content_hash FROM atoms a JOIN atom_versions v USING(atom_id) WHERE v.is_latest'):
  catalog.setdefault(r['fqdn'],[]).append(r)
 deps=db.execute("SELECT d.dependency_artifact_fqdn,d.dependency_content_hash,d.optional,d.dependency_role,a.artifact_id FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id JOIN catalog_artifacts_served a ON a.artifact_id=v.artifact_id WHERE v.is_latest AND a.artifact_kind='cdg'").fetchall()
 totals=Counter();reports=[]
 for row in rows:
  snapshot=row['details']['snapshot']
  if hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=row['content_hash']:raise ValueError('Snapshot drift')
  bindings={b['stage_id']:b for b in snapshot['bindings']['bindings']};outcomes=Counter();states=Counter()
  for stage in snapshot['template']['stages']:
   b=bindings.get(stage['stage_id'],{});states[b.get('status','missing')]+=1
   target=b.get('bound_artifact_fqdn');candidates=catalog.get(target,[])
   if not target:outcome='no_named_implementation'
   elif len(candidates)!=1:outcome='missing_or_ambiguous_current_catalog_version'
   else:
    c=candidates[0];runtime=(c['import_module'] or '')+'.'+(c['source_symbol'] or '')
    fn=REGISTRY.get(runtime,{}).get('impl')
    if fn is None:outcome='current_callable_not_registered'
    else:
     contract=assess_keyword_call_contract([p['name'] for p in stage.get('inputs',[])],fn)
     outcome='keyword_compatible' if contract['keyword_compatible'] else 'adapter_or_missing_computation'
   outcomes[outcome]+=1
  links={str(d['artifact_id']) for d in deps if d['dependency_artifact_fqdn']==row['fqdn'] and d['dependency_content_hash']==row['content_hash']}
  mandatory_links={str(d['artifact_id']) for d in deps if d['dependency_artifact_fqdn']==row['fqdn'] and d['dependency_content_hash']==row['content_hash'] and not d['optional'] and d['dependency_role']=='cdg'}
  totals.update(outcomes)
  current=selected_catalog_state(db,row)
  reports.append(dict(version_id=str(row['version_id']),stages=len(snapshot['template']['stages']),outcomes=dict(outcomes),binding_states=dict(states),original_status=row['status'],original_publishable=row['is_publishable'],served_realizations_with_direct_provenance=len(links),served_realizations_with_mandatory_direct_provenance=len(mandatory_links),selected_catalog_version=current))
 report=dict(read_only=True,templates=len(rows),stage_outcomes=dict(totals),templates_with_direct_served_provenance=sum(bool(r['served_realizations_with_direct_provenance']) for r in reports),templates_all_keyword_compatible=sum(r['outcomes'].get('keyword_compatible',0)==r['stages'] for r in reports),templates_detail=reports,scope='Current callable/keyword coverage and direct served source-provenance only. A linked realization may be partial; no original-template completion or approval claim. Dataset content and identifying metadata excluded.')
 selected=[r['selected_catalog_version'] for r in reports]
 report['original_artifacts_currently_served']=sum(r['served'] for r in selected)
 report['original_artifacts_with_served_corrected_version']=sum(r['served'] and r['corrected_version'] for r in selected)
 report['served_versions_with_execution_envelope_and_exact_approved_bindings']=sum(
  r['served'] and r['execution_envelope_valid'] and r['active_bindings_complete'] and r['approved_provider_versions_exact']
  and r['passed_version_audit'] and not r['failed_version_audits'] for r in selected)
 report['scope']='Historical intake keyword coverage, direct source provenance and separately reported current-version catalog integrity. A served corrected version does not make its historical placeholder interfaces executable. This audit does not rerun semantic/source/runtime qualification or establish completion of an entire source solution. Dataset content and identifying metadata excluded.'
 (root/'docs/reviews/competition_current_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='templates_detail'},indent=2))
 ranked=sorted(reports,key=lambda r:(-r['outcomes'].get('keyword_compatible',0),-r['outcomes'].get('adapter_or_missing_computation',0),r['stages']))
 print('ranked',json.dumps(ranked[:8],indent=2))
