#!/usr/bin/env python3
"""Read-only inventory of remaining physics graph evidence and expression gates.

This audit diagnoses stored evidence. It does not refresh source comparison,
execute proofs, or grant publication approval.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts, contract_blockers


def audit(root, rule_file):
    rule_bytes = rule_file.read_bytes()
    reports = []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graphs = db.execute("SELECT a.artifact_id,a.fqdn,v.version_id,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_kind='cdg' AND a.fqdn LIKE 'physics.pdg.%' AND a.status='draft' AND v.is_latest ORDER BY a.fqdn").fetchall()
        for graph in graphs:
            nodes = db.execute('SELECT type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (graph['version_id'],)).fetchall()
            signatures = [json.loads(n['type_signature']) if isinstance(n['type_signature'],str) and n['type_signature'].strip() else (n['type_signature'] or {}) for n in nodes]
            bindings = db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s', (graph['version_id'],)).fetchall()
            blockers, rules_used = Counter(), Counter()
            eligible = 0
            pins = set()
            for binding in bindings:
                if binding['status'] != 'active':
                    blockers['inactive_expression_binding'] += 1
                    continue
                candidates = db.execute("SELECT e.evidence_json,s.payload FROM artifacts a JOIN artifact_versions ov ON ov.artifact_id=a.artifact_id JOIN artifact_versions v ON v.artifact_id=a.artifact_id AND (v.version_id=ov.version_id OR v.derives_from=ov.version_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE a.fqdn=%s AND ov.content_hash=%s", (binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
                for c in candidates:
                    pin=c['payload'].get('core_file_sha256',{}).get('conversion_of_data_formats/infrules.cypher')
                    if pin: pins.add(pin)
                dimensioned = [c for c in candidates if c['evidence_json'].get('dimensional_analysis',{}).get('status') == 'passed']
                exact = [c for c in dimensioned if c['evidence_json'].get('pdg_source_comparison',{}).get('source_comparison',{}).get('correspondence') == 'exact_ast_match' or c['evidence_json'].get('source_identity_comparison',{}).get('correspondence') == 'exact_source_identity_ast']
                if not candidates: blockers['missing_expression_snapshot'] += 1
                elif not dimensioned: blockers['no_dimensionally_valid_expression_version'] += 1
                elif not exact: blockers['no_exact_source_correspondence'] += 1
                elif len(dimensioned) != 1 or len(exact) != 1: blockers['ambiguous_expression_version'] += 1
                else: eligible += 1
            source_steps = bool(signatures) and all(s.get('source_pdg_step_id') for s in signatures)
            if not source_steps: blockers['requires_source_step_projection'] += 1
            if len(pins) == 1:
                rules = load_pinned_rule_contracts(rule_bytes,next(iter(pins)))
                for signature in signatures:
                    rule = rules.get(signature.get('inference_rule_id'))
                    if rule is None: blockers['missing_rule_contract'] += 1
                    else:
                        rules_used[rule.name] += 1
                        blockers.update(contract_blockers(signature,rule))
            else: blockers['missing_or_ambiguous_rule_pin'] += 1
            evidence = db.execute("SELECT e.runner_version,e.passed,e.details->>'blocker' AS blocker, "
                "e.version_id::text AS evidence_version_id,ev.derives_from::text AS derives_from,ev.is_latest AS evidence_version_is_latest "
                "FROM artifact_audit_evidence e JOIN artifact_versions ev ON ev.version_id=e.version_id "
                "WHERE e.artifact_id=%s AND e.runner_version IN ('pdg-graph-replay.v1','pdg-source-graph-replay.v2','pdg-graph-replay.v3') "
                "ORDER BY e.runner_version,e.passed,e.version_id", (graph['artifact_id'],)).fetchall()
            realizations = db.execute("SELECT count(DISTINCT a.artifact_id) AS n FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id JOIN catalog_artifacts_served a USING(artifact_id) WHERE d.dependency_artifact_fqdn=%s AND d.dependency_role='cdg' AND a.artifact_kind='cdg' AND v.is_latest", (graph['fqdn'],)).fetchone()['n']
            reports.append(dict(artifact_id=str(graph['artifact_id']),version_id=str(graph['version_id']),content_hash=graph['content_hash'],
                source_step_projection=source_steps,nodes=len(nodes),bindings=len(bindings),eligible_expression_bindings=eligible,
                blockers=dict(sorted(blockers.items())),inference_rules=dict(sorted(rules_used.items())),
                recorded_replays=evidence,served_derived_realizations=realizations))
    totals=Counter()
    for r in reports: totals.update(r['blockers'])
    return dict(read_only=True,approval_applied=False,draft_graphs=len(reports),
        source_step_graphs=sum(r['source_step_projection'] for r in reports),
        graphs_with_complete_stored_expression_gates=sum(bool(r['bindings']) and r['eligible_expression_bindings']==r['bindings'] for r in reports),
        blocker_counts=dict(sorted(totals.items())),graphs=reports,
        rule_file_sha256=hashlib.sha256(rule_bytes).hexdigest(),
        limitations=['Stored evidence only, including explicitly separate source-ID alternatives; source bytes, implementation hashes and execution must be freshly verified before approval.',
            'A served derived realization does not approve the original conceptual graph or imply complete semantic scope.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(Path(__file__).resolve().parents[1],args.rule_file)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='graphs'}))
