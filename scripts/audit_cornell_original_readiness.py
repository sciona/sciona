"""Reconcile retained Cornell approval with current native publication requirements."""
import json
from pathlib import Path
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.restore_cornell_reviewed_software import sha as bytes_sha, verify
from sciona.cornell_augmentation import manifest, PINS
SOURCE_VERSION = "5326bafb-d26f-5202-85c9-6f903df0fc77"

def sha(path): return bytes_sha(Path(path).read_bytes())


def audit():
    verified = verify(Path('/private/tmp/sciona_cornell_source'),
                      manifest('competition_cornell_source_pins.json', PINS)['pins'])
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        original = db.execute('SELECT a.fqdn FROM artifacts a JOIN artifact_versions v USING(artifact_id) '
                              'WHERE version_id=%s', (SOURCE_VERSION,)).fetchone()['fqdn']
        names = [original, original + '.execution', 'sciona.atoms.ml.cornell_execution.cornell_prepare',
                 'sciona.atoms.ml.cornell_execution.cornell_execute']
        for name in names:
            rows = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.artifact_kind,a.status,a.is_publishable,
                v.version_id::text,v.content_hash,v.trust_tier,artifact_is_publishable(a.artifact_id) AS eligible,
                EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
                FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest''',
                (name,)).fetchall()
            if len(rows) != 1:
                raise ValueError('Unique current Cornell catalog identity required')
            row = rows[0]
            catalogs = {}
            for prefix, key in [('artifact', 'artifact_id'), ('atom', 'atom_id')]:
                if prefix == 'atom' and row['artifact_kind'] != 'atom':
                    continue
                counts = {}
                for suffix in ('descriptions', 'parameters', 'references', 'audit_rollups'):
                    counts[suffix] = db.execute(f'SELECT count(*) AS n FROM {prefix}_{suffix} WHERE {key}=%s',
                                               (row['artifact_id'],)).fetchone()['n']
                descriptions = db.execute(f'SELECT kind,language,jargon_score FROM {prefix}_descriptions '
                                          f'WHERE {key}=%s ORDER BY kind,language', (row['artifact_id'],)).fetchall()
                ports = db.execute(f'SELECT direction,name,type_desc,ordinal,required,default_value_repr,constraints '
                                   f'FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                                   (row['version_id'],)).fetchall()
                eligible = db.execute(f'SELECT {prefix}_is_publishable(%s) AS eligible',
                                      (row['artifact_id'],)).fetchone()['eligible']
                catalogs[prefix] = dict(counts=counts, descriptions=descriptions, ports=ports, eligible=eligible)
            records.append(dict(identity=row, catalogs=catalogs))
    execution = json.loads((ROOT/'docs/reviews/cornell_current_graph.json').read_text())
    if execution['passed'] is not True or (execution['models_completed'], execution['training_phases'], execution['continuations']) != (13,17,4):
        raise ValueError('Full current graph execution required')
    for path, digest in execution['software_sha256'].items():
        if sha(ROOT/path) != digest: raise ValueError('Executed software changed')
    return dict(format='cornell-original-readiness.v1', read_only=True, catalog_mutations=0,
        original_ready_for_promotion=False, records=records,
        source_files_verified=verified,
        current_local_graph_execution_verified=True,
        execution_sha256=sha(ROOT/'docs/reviews/cornell_current_graph.json'),
        auditor_sha256=sha(Path(__file__)),
        next_requirements=[
            'Review reusable provider contracts and add missing plain descriptions and parameter metadata with transaction gates.',
            'Require both native eligibility predicates, exact runtime bindings and current semantic bounds for reused atoms.',
            'Review original five-stage scope and reusable operation boundaries before creating its corrected version.',
            'Execute the corrected stored original, then test transactional publication and production selection.'],
        limits=[
            'Retained served flags alone do not establish current native publication eligibility.',
            'Local serialized full-model execution is distinct from retained catalog graph execution and original source-scope review.',
            'No new approval, historical competition accuracy, cross-domain accuracy or human Tier 1 review is claimed.'])


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/cornell_original_readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(original_ready=False, records=len(report['records']),
                         native_eligible=sum(r['identity']['eligible'] for r in report['records']),
                         source_files_verified=report['source_files_verified'], catalog_mutations=0)))
