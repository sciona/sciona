"""Read-only lineage, software and generic-reduction preflight; no promotion."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path

import numpy as np
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS, predict_twenty
from sciona.openvaccine_models import SOURCE_SHA256, LICENSE_SHA256
from sciona.openvaccine_folding import BINARY_SHA256, PARAMETERS_SHA256
from scripts.import_flavours_provider_drafts import connect

ROOT = Path(__file__).resolve().parents[1]
ORIGINALS = ('d132e834-0634-59a6-adc5-2fe3b00082e3', '26cf554f-1920-5652-853d-99b62dc3aade')
DERIVATIVE = '19cf56e7-8fc1-5dad-ac8a-51d58c08f6e9'
GENERIC = 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduction_check():
    rng = np.random.default_rng(681254)
    cases = 0
    for batch in (1, 3):
        for length in (2, 7, 14):
            nodes = rng.normal(size=(batch, length+2, 55)).astype(np.float32)
            adjacency = rng.normal(size=(batch, length+2, length+2, 8)).astype(np.float32)
            values = rng.uniform(-20, 20, size=(20, batch, length, 5)).astype(np.float32)
            class Fixed:
                def __init__(self, family, value):
                    self.family, self.value = family, value
                def predict(self, n, a):
                    return self.value.copy()
            for order in (np.arange(20), np.arange(20)[::-1], rng.permutation(20)):
                for ddof in (0, 1):
                    members = [(family, slot, Fixed(family, values[i]))
                               for i, (family, slot) in enumerate(sorted(EXPECTED_MEMBERS))]
                    ordered = [members[int(i)] for i in order]
                    saved = nodes.copy(), adjacency.copy(), values.copy()
                    expected = predict_twenty(ordered, nodes, adjacency, std_ddof=ddof)
                    # Preserve native float32 forward/reverse averaging before
                    # float64 clipping, ordered addition, division and final cast.
                    rows = np.stack([np.clip(((values[i]+values[i, :, ::-1, :])/2).astype(np.float64),
                                             -.5, 6.).reshape(-1) for i in order])
                    total = ordered_weighted_sum(rows, np.ones(20))
                    actual = (total.reshape(batch, length, 5)/20).astype(np.float32)
                    if actual.tobytes() != expected['prediction'].tobytes():
                        raise ValueError('Generic reduction changes ensemble arithmetic')
                    if any(a.tobytes() != b.tobytes() for a,b in zip(saved,(nodes,adjacency,values))):
                        raise ValueError('Reduction changed caller inputs')
                    cases += 1
    return dict(cases=cases, bitwise_prediction_equality=True, inputs_unchanged=True,
        members=20, member_orders=3, teacher_ddof_values=[0,1], synthetic_only=True,
        scope='Fixed synthetic model outputs exercise the unchanged ensemble oracle. This verifies final clipped reduction only; no neural training, teacher-moment extraction or folding execution claim.')


def audit():
    old = json.loads((ROOT/'docs/reviews/competition_openvaccine_graph_execution.json').read_text())
    changed = [name for name,digest in old['code_sha256'].items() if sha(ROOT/name) != digest]
    versions = dict(line.split('==') for line in (ROOT/'requirements/openvaccine-execution.txt').read_text().splitlines() if line)
    drift = {}
    for name, version in versions.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            drift[name] = dict(recorded=version, installed=installed)
    software = {
        'model_definitions': (Path('/private/tmp/sciona_openvaccine_source/scripts/nullrecurrent_inference.py'), SOURCE_SHA256),
        'model_license': (Path('/private/tmp/sciona_openvaccine_source/LICENSE'), LICENSE_SHA256),
        'folding_executable': (Path('/private/tmp/sciona_openvaccine_contrafold/src/contrafold'), BINARY_SHA256),
        'folding_coefficients': (Path('/private/tmp/sciona_openvaccine_eternafold_parameters/parameters/EternaFoldParams.v1'), PARAMETERS_SHA256),
    }
    provisioned = {name:dict(available=path.is_file(), matches_reviewed_hash=path.is_file() and sha(path)==digest)
                   for name,(path,digest) in software.items()}
    provider = ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/openvaccine_execution.py'
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        catalog = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE v.is_latest AND a.artifact_id::text=ANY(%s) ORDER BY a.fqdn''', (list(ORIGINALS)+( [DERIVATIVE]),)).fetchall()
        if len(catalog) != 3:
            raise ValueError('Both source identities and existing derivative required')
        dependencies = db.execute('''SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional
            FROM artifact_dependencies WHERE dependent_version_id IN
            (SELECT version_id FROM artifact_versions WHERE artifact_id::text=%s AND is_latest)
            ORDER BY dependency_artifact_fqdn''', (DERIVATIVE,)).fetchall()
        for original in (row for row in catalog if row['artifact_id'] in ORIGINALS):
            if not any(d['dependency_artifact_fqdn']==original['fqdn'] and d['dependency_content_hash']==original['content_hash']
                       and d['dependency_role']=='cdg' and d['optional'] is False for d in dependencies):
                raise ValueError('Existing derivative lost mandatory original provenance')
        providers = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,artifact_is_publishable(a.artifact_id) AS native_eligible,
            (SELECT count(*) FROM artifact_descriptions d WHERE d.artifact_id=a.artifact_id) AS descriptions,
            (SELECT count(*) FROM artifact_parameters p WHERE p.version_id=v.version_id) AS parameters
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.is_latest
            AND a.fqdn=ANY(%s) ORDER BY a.fqdn''',
            (['sciona.atoms.ml.openvaccine_execution.openvaccine_prepare',
              'sciona.atoms.ml.openvaccine_execution.openvaccine_execute', GENERIC],)).fetchall()
    return dict(format='decomposed-openvaccine-readiness.v1', read_only=True,catalog_mutations=0,approved=False,
        catalog=catalog,derivative_dependencies=dependencies,providers=providers,
        prior_execution_changed_files=changed,prior_provider_hash_matches=sha(provider)==old['provider_sha256'],
        runtime_drift=drift,provisioned_software=provisioned,synthetic_reduction=reduction_check(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'sciona').glob('openvaccine_*.py'))},
        generic_provider_sha256=sha(inspect.getfile(ordered_weighted_sum)),auditor_sha256=sha(__file__),
        next_steps=[
            'Verify provisioned model source, compiler build and folding coefficients before native execution; preserve historical reports.',
            'Preserve both original source histories and the shared derivative. Review the two original scopes explicitly before approval; reuse shared providers rather than create duplicate implementations.',
            'Separate raw RNA folding/features from feature-level model training and refinement. Document which interfaces admit caller-defined graph features and which retain RNA assumptions.',
            'Retain full autoencoder topology, all twenty models, explicit supervised splits, every caller-specified refinement round, checkpoint restoration and teacher statistics.',
            'Expose final clipped members for the domain-independent ordered sum, preserving float32 reverse averaging, float64 clipping/addition and final float32 mean. Keep teacher moments unchanged.',
            'Revalidate synthetic execution under one-thread native and TensorFlow pools, then qualify provider metadata, publication rollback and original-CDG production selection.'],
        limits=['Readiness only; no new original or provider approval.',
                'Generic sum reuse does not make RNA folding domain independent. Dense graph features require matching shape and semantic assumptions.',
                'Twenty-member reconstruction is not proof of the original winning clustered blend or historical optimization schedule.',
                'Package metadata comparison does not verify actual imports, installed package files, native execution or clean installation.'])


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/decomposed_openvaccine_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('prior_execution_changed_files','prior_provider_hash_matches',
        'runtime_drift','provisioned_software','synthetic_reduction')}))
