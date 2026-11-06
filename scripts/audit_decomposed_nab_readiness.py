"""Read-only current NAB catalog prerequisites; preserve historical evidence."""
import hashlib
import importlib.metadata as metadata
import json
from scripts.import_flavours_provider_drafts import ROOT, connect
from sciona.decomposed_nab_graph import build_decomposed_nab_graph


def audit():
    requirements = ROOT/'requirements/nab-execution.txt'
    expected = dict(line.split('==') for line in requirements.read_text().splitlines() if line)
    differences = []
    for package, version in expected.items():
        actual = metadata.version(package)
        if actual != version:
            differences.append(dict(package=package, historical=version, current=actual))
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,
            EXISTS (SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.artifact_id=(SELECT artifact_id FROM artifact_versions WHERE version_id=%s)
            AND v.is_latest''', ('e78e1ac4-49ad-5fe2-9a58-c82862d11ec7',)).fetchall()
        providers = []
        for node in build_decomposed_nab_graph().nodes:
            rows = db.execute('''SELECT a.status,a.is_publishable,v.version_id::text,
                v.content_hash,v.trust_tier,artifact_is_publishable(a.artifact_id) AS native_eligible
                FROM artifacts a JOIN artifact_versions v USING(artifact_id)
                WHERE a.fqdn=%s AND v.is_latest''', (node.matched_primitive,)).fetchall()
            providers.append(dict(node=node.node_id, fqdn=node.matched_primitive, versions=rows))
    return dict(format='decomposed-nab-readiness.v1', catalog_mutations=0, read_only=True,
        original_latest=original, historical_package_differences=differences, providers=providers,
        current_graph_execution_sha256=hashlib.sha256((ROOT/'docs/reviews/decomposed_nab_graph_execution.json').read_bytes()).hexdigest(),
        next_requirements=['Revalidate pinned benchmark scoring software and dependency closure on the current environment.',
            'Review reusable threshold and recipe provider semantics, fingerprints, interfaces and negative gates.',
            'Stage and approve exact provider versions, preserving existing history.',
            'Stage the complete corrected graph under the original identity, run it from storage and validate publication rollback gates.',
            'Approve at automated Tier 3 and verify exact production selection and idempotency.'],
        limits=['The source intake describes a generic isolation-forest or autoencoder choice; the implemented branch is isolation forest.',
                'Current graph execution proves synthetic implementation equivalence, not historical detector or accuracy parity.'],
        auditor_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/decomposed_nab_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
