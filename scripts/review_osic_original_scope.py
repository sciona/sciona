"""Review pinned source code and a synthetic head counterexample; no promotion.

Notebook outputs are never read into the report. The source checkout remains
private runtime material; only software identities and review findings are saved.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciona.competition_import import _canonical, _id
from scripts.import_flavours_provider_drafts import ROOT, connect

FQDN = 'cdg.competition.solution.kaggle.osic_pulmonary_1st'
VERSION = '0a53a43f-b312-5b74-952d-496366e623ad'
DIGEST = 'cdc68a172f66f95775d6263c4ba8563174f7b652fc84d961105a49518a6c78c7'
REPOSITORY = 'https://github.com/artkulak/osic-pulmonary-fibrosis-progression'
REVISION = 'cd6dfa56c012e3c0eb81ed15534f436b3ba16d19'
TRAIN = 'effnet_b5_training_30_epochs.ipynb'
INFERENCE = 'inference-45-55-600-epochs-tuned-effnet-b5-30-ep.ipynb'
PINS = {
    'get_efficientnet': '4d5bc39e4283b05f54fa1851d080a1043a2a58a1f63b23afe3aa8ef77bd9f974',
    'build_model': '29a0d3d7730e3f91f5bbc92e4d483892e95dee1b4b7458f9fc30403cd016db69',
    'score': 'a7846c4cccd5c6260345fa89b88a1b1589d3f9a0186898cfc41bec91c9623434',
    'qloss': 'eb775c738637dac1b91f6bbca81aa72ce4db212a06a8adffb366b4b53aab5542',
    'mloss': '23f320fc7f033713d72d674c7db3073be2f67eb7695f7db61cf5964e608ec001',
    'make_model': '620c9a7a197f003d81a5914e45e9703faaa24bb00bc79df225c4fae6428558b0',
}
CORRECTIONS = {
    'image_feature_extraction': 'The source trains an EfficientNet-B5 slope predictor from random weights with a tabular side input, noise and dropout. It is not a frozen pretrained feature extractor for the quantile branch.',
    'tabular_feature_engineering': 'Keep branch-specific feature preparation, longitudinal response construction, population selection and preprocessing state explicit. The source removes a previously used covariate; generic baseline/delta descriptions do not specify the full preparation.',
    'feature_concatenation': 'Concatenation occurs inside the image predictor. The final solution blends two separate predictors; the intake incorrectly routes extracted image features into the quantile model.',
    'quantile_regression_training': 'The source uses TensorFlow/Keras, two hidden dense layers, a mixed quantile and clipped likelihood loss, two folds and 600 epochs. An unconstrained base head plus cumulative nonnegative increments does not guarantee ordered predictions.',
    'confidence_interval_postprocessing': 'Qualify both branches and the final prediction/confidence blend. A single quantile-spread conversion does not cover the complete source computation; spread is also not proof of calibrated coverage.',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def review(source_directory):
    directory = Path(source_directory)
    functions = {}
    for filename, cell, names in [(TRAIN, 20, ('get_efficientnet', 'build_model')),
                                  (INFERENCE, 32, ('score', 'qloss', 'mloss', 'make_model'))]:
        notebook = json.loads((directory/filename).read_text())
        source = ''.join(notebook['cells'][cell]['source'])
        definitions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        for name in names:
            node = definitions[name]
            if sha(ast.unparse(node).encode()) != PINS[name]:
                raise ValueError('Pinned source function changed: '+name)
            functions[name] = node
    schedule = {}
    for filename, cell, wanted in [(TRAIN, 11, {'EPOCHS': 30, 'BATCH_SIZE': 8}),
                                   (INFERENCE, 35, {'NFOLD': 2}),
                                   (INFERENCE, 36, {'EPOCHS': 600})]:
        notebook = json.loads((directory/filename).read_text())
        source = '\n'.join(line for line in ''.join(notebook['cells'][cell]['source']).splitlines()
                           if not line.lstrip().startswith(('!', '%')))
        found = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        found[target.id] = ast.literal_eval(node.value)
        if found != wanted:
            raise ValueError('Source training schedule changed')
        schedule[('image' if filename == TRAIN else 'quantile')+'.cell'+str(cell)] = found
    license_digest = sha((directory/'LICENSE').read_bytes())
    if license_digest != '6b0382b16279f26ff69014300541967a356a666eb0b91b422f6862f6b7dad17e':
        raise ValueError('Source license changed')
    readme_digest = sha((directory/'README.md').read_bytes())
    if readme_digest != 'e3650e83eb8d2be425873869f291bbc365e8123f0aa9cdfd5bf21023963f5694':
        raise ValueError('Pinned author writeup changed')

    # Execute only the reviewed numerical lambda, not notebook code or outputs.
    heads = [node for node in ast.walk(functions['make_model']) if isinstance(node, ast.Lambda)]
    if len(heads) != 1 or ast.unparse(heads[0]) != 'lambda x: x[0] + tf.cumsum(x[1], axis=1)':
        raise ValueError('Source prediction head changed')
    expression = ast.Expression(body=heads[0])
    head = eval(compile(ast.fix_missing_locations(expression), '<reviewed-source-head>', 'eval'),
                {'__builtins__': {}, 'tf': SimpleNamespace(cumsum=np.cumsum)})
    base = np.array([[3., 2., 1.], [0., 0., 0.]])
    increments = np.array([[0., 0., 0.], [1., 2., 3.]])
    result = head([base, increments])
    np.testing.assert_array_equal(result, np.array([[3., 2., 1.], [1., 3., 6.]]))
    np.testing.assert_array_equal(np.all(np.diff(result, axis=1) >= 0, axis=1), [False, True])
    np.testing.assert_array_equal(base, [[3., 2., 1.], [0., 0., 0.]])
    np.testing.assert_array_equal(increments, [[0., 0., 0.], [1., 2., 3.]])

    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        rows = db.execute("""SELECT e.details FROM artifact_audit_evidence e
            JOIN artifacts a USING(artifact_id) WHERE a.fqdn=%s AND e.version_id=%s
            AND e.runner_version='competition-intake.v1'""", (FQDN, VERSION)).fetchall()
        if len(rows) != 1:
            raise ValueError('Unique original intake required')
        snapshot = rows[0]['details']['snapshot']
        if sha(_canonical(snapshot).encode()) != DIGEST or _id(FQDN+'/'+DIGEST) != VERSION:
            raise ValueError('Original intake identity changed')
        stages = snapshot['template']['stages']
        if len(stages) != 5 or {s['stage_id'] for s in stages} != set(CORRECTIONS):
            raise ValueError('Original scope changed')
    return dict(format='osic-original-source-scope.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, fqdn=FQDN,
        source_version_id=VERSION, source_content_hash=DIGEST,
        primary_source=dict(repository=REPOSITORY, revision=REVISION,
            software_paths=[TRAIN, INFERENCE], normalized_function_sha256=PINS,
            readme_sha256=readme_digest, license_sha256=license_digest,
            checked_schedule_constants=schedule,
            declared_license='Unlicense', permission_scope='Repository declaration only; external dependencies and model artifacts require separate review.'),
        stage_review=[dict(stage_id=s['stage_id'], correction=CORRECTIONS[s['stage_id']]) for s in stages],
        synthetic_head_check=dict(source_expression_executed=True, cases=2,
            crossing_counterexample_confirmed=True, ordered_control_confirmed=True,
            inputs_unchanged=True, backend='NumPy evaluation of pinned cumulative-sum expression; not TensorFlow training qualification'),
        reuse_requirements=[
            'Separate image preprocessing, tabular preparation, slope fitting, trajectory extrapolation, quantile fitting, uncertainty calculation and final blending.',
            'Expose coordinates, time units, prediction availability, image geometry, random state, fitted transforms and blend coefficients.',
            'Preserve source output behavior, including possible crossings; do not silently sort predictions or claim guaranteed interval coverage.',
            'Keep clinical response construction and cohort policies in adapters; demonstrate numerical reuse on distinct synthetic applications.'],
        remaining=[
            'Pin and inspect every executable cell needed for training and inference, including branch-specific preprocessing and final blending.',
            'Resolve historical dependency semantics, especially optimizer decay, image library behavior and random initialization.',
            'Preserve the complete 30-epoch image schedule and two-fold 600-epoch quantile schedule with bounded resources.',
            'Qualify reusable operations and domain adapters, then execute the complete corrected graph with synthetic inputs.',
            'Pass dependency, metadata, rollback and served-selection gates under the original intake identity.'],
        limitations=['Source review and synthetic head counterexample only; no complete workflow, clinical validity or historical accuracy claim.',
            'No source notebook outputs or real records are included in this report.',
            'Quantile-head arithmetic is checked with NumPy; TensorFlow numerical and gradient parity remain unqualified.'],
        reviewer_sha256=sha(Path(__file__).read_bytes()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    result = review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_original_source_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, source_stages=len(result['stage_review']),
                         synthetic_head_check=result['synthetic_head_check'], approved=False)))
