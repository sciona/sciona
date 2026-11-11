"""Read-only prerequisites for correcting the original Santa CDG identity."""
import hashlib
import importlib.metadata as metadata
import json
from scripts.import_flavours_provider_drafts import ROOT, connect
from sciona.decomposed_santa_graph import build_decomposed_santa_graph


def audit():
    expected = dict(line.split('==') for line in
                    (ROOT/'requirements/santa-execution.txt').read_text().splitlines() if line)
    differences = [dict(package=name,historical=version,current=metadata.version(name))
                   for name,version in expected.items() if metadata.version(name) != version]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original = db.execute('''SELECT a.artifact_id::text,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,
            EXISTS (SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.artifact_id=(SELECT artifact_id FROM artifact_versions WHERE version_id=%s)
            AND v.is_latest''', ('8ab2dc68-65da-5415-bf3a-bbebde6fe898',)).fetchall()
        providers = []
        for node in build_decomposed_santa_graph().nodes:
            rows = db.execute('''SELECT a.status,a.is_publishable,v.version_id::text,
                v.content_hash,v.trust_tier,artifact_is_publishable(a.artifact_id) AS native_eligible
                FROM artifacts a JOIN artifact_versions v USING(artifact_id)
                WHERE a.fqdn=%s AND v.is_latest''', (node.matched_primitive,)).fetchall()
            providers.append(dict(node=node.node_id,fqdn=node.matched_primitive,versions=rows))
    return dict(format='decomposed-santa-readiness.v1',catalog_mutations=0,read_only=True,
        original_latest=original,historical_package_differences=differences,providers=providers,
        next_requirements=[
            'Revalidate source correction and installed dependency closure on the current environment.',
            'Review all explicit stages and exact source-computation parity, including negative and extrapolating blend coefficients.',
            'Qualify exact providers and preserve existing histories; do not relabel historical evidence.',
            'Stage and execute the complete graph under the original identity with mandatory source provenance.',
            'Validate rollback, approve at Tier 3, verify production selection and zero-change repeat.'],
        limits=['Original Beta/UCB intake is contradicted by the recovered winner implementation; the correction must remain explicit.',
                'Two replay-trained regressors and all causal feature computation are retained; historical population and competitive accuracy remain unproven.'],
        auditor_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/decomposed_santa_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
