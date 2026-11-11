"""Reconcile retained HuBMAP approval with current native publication requirements."""
import json
from pathlib import Path
from scripts.import_flavours_provider_drafts import ROOT, connect
import hashlib

def bytes_sha(value): return hashlib.sha256(value).hexdigest()
SOURCE_VERSION = "325c30ff-ea2a-515c-8d40-110a5fd6991d"

def sha(path): return bytes_sha(Path(path).read_bytes())


def audit():
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        original = db.execute('SELECT a.fqdn FROM artifacts a JOIN artifact_versions v USING(artifact_id) '
                              'WHERE version_id=%s', (SOURCE_VERSION,)).fetchone()['fqdn']
        from sciona.hubmap_execution import build_hubmap_lifecycle_graph
        graph = build_hubmap_lifecycle_graph()
        names = [original, original + '.execution', *sorted({n.matched_primitive for n in graph.nodes})]
        for name in names:
            rows = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.artifact_kind,a.status,a.is_publishable,
                v.version_id::text,v.content_hash,v.trust_tier,artifact_is_publishable(a.artifact_id) AS eligible,
                EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
                FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest''',
                (name,)).fetchall()
            if len(rows) != 1:
                raise ValueError('Unique current HuBMAP catalog identity required')
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
    execution_path = ROOT/'docs/reviews/competition_hubmap_serialized.json'
    execution = json.loads(execution_path.read_text())
    drift = [path for path,digest in execution['implementation_sha256'].items() if sha(ROOT/path) != digest]
    provider_path = ROOT.parent/'sciona-atoms-dl/src/sciona/atoms/dl/segmentation/hubmap_execution.py'
    provider_unchanged = sha(provider_path) == execution['provider_sha256']
    return dict(format='hubmap-original-readiness.v1', read_only=True, catalog_mutations=0,
        original_ready_for_promotion=False, records=records,
        historical_execution_software_drift=drift,
        historical_provider_unchanged=provider_unchanged,
        historical_execution_checks={k:execution[k] for k in ('synthetic_only','independent_raw_lifecycle_reference','graph_nodes','graph_edges','exact_both_stage_metrics_checkpoints_rng','exact_generated_pseudo_and_final_masks')},
        execution_sha256=sha(execution_path),
        full_model_coverage='Historical serialized evidence covers six connected nodes, one fold and one epoch in each training stage, compared to an independent raw lifecycle reference. Current-runtime and source-sized qualification remain required.',
        auditor_sha256=sha(Path(__file__)),
        next_requirements=[
            'Review reusable provider contracts and add missing plain descriptions and parameter metadata with transaction gates.',
            'Require both native eligibility predicates, exact runtime bindings and current semantic bounds for reused atoms.',
            'Review every original source stage and reusable operation boundaries before creating its corrected version.',
            'Qualify numerical, encoder, augmentation and image-codec dependencies; execute initial training, model restoration, pseudo masks, fresh retraining and final inference with justified fold coverage and geometry.',
            'Execute the corrected stored original, then test transactional publication and production selection.'],
        limits=[
            'Retained served flags alone do not establish current native publication eligibility.',
            'Historical full-ensemble evidence is distinct from current-runtime and corrected-original qualification.',
            'No new approval, historical competition accuracy, cross-domain accuracy or human Tier 1 review is claimed.'])


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/hubmap_original_readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(original_ready=False, records=len(report['records']),
                         native_eligible=sum(r['identity']['eligible'] for r in report['records']),
                         software_drift=len(report['historical_execution_software_drift']), catalog_mutations=0)))
