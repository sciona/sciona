"""Automated Tier 3 review of the complete generic biosignal-sequence topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='1587d081-d47c-5595-8bed-d3d3aa6465af'
SOURCE_HASH='2ab7a80c92296d3fffeb6187655e52443caf311086b4cdc3ed1efde5fe4406c8'
SCOPE='Generic five-stage biosignal spectrogram/sequence topology; independent binary waveform/spectrogram realization.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft with mandatory provenance.', 'Complete five-stage binary waveform/spectrogram realization with spectral masking and recording-level threshold decisions. Independent project architecture; synthetic numerical and lifecycle qualification only, no clinical or historical accuracy claim.', 'Caller supplies finite channel/time arrays with shared sample rate and consistent channel meaning/order. Subject identifiers must be meaningful and disjoint across training/calibration/query. No automatic subject, montage, unit or channel-identity discovery.', 'Fixed windows never cross recording boundaries. Incomplete final windows and FFT segments are omitted. Spectral features use segment-mean detrending, periodic Hann, one-sided density-normalized PSD and log1p; no learned spectral statistics across populations.', 'Training-only channel scaling gives recordings equal total weight over included windows; constant scales become one. Overlapping windows repeat samples in statistics. Class-balanced recording loss gives recordings equal weight within each class, not equal subject weight.', 'CPU float64 waveform 1D and spectrogram 2D CNNs, with temporal pooling retaining frequency positions. Recording labels supervise every included window. Seeded time/frequency masks zero normalized spectral bins; waveform inputs remain unmasked.', 'Adam uses globally weighted gradient accumulation with one update per epoch. No early stopping or validation checkpoint selection. All features stored in memory; batching limits neural forward allocation, not total feature memory.', 'Mean window probability yields each recording score. Separate both-class calibration chooses maximal binary F1 with highest-threshold tie rule. Query labels rejected. Calibration selection is not unbiased accuracy; no probability calibration guarantee.', 'Synthetic tests, pinned provisioned runtime dependencies and installed notices only. No clean-install, cross-platform, GPU, real-data or resource qualification. Strict JSON boundary and trusted in-process intermediates; no portable model serialization claim.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_biosignal_sequence_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==34 and tests['serialized_boundary_tests']==13
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'subject_safe_windowing': 'Caller-supplied disjoint subjects, aligned channels/sample rate, fixed within-recording windows', 'spectral_signal_features': 'Periodic Hann detrended one-sided density PSD with log1p and training-only balanced channel scaling', 'deep_signal_backbones': 'Joint 1D waveform and 2D spectrogram CNNs with frequency-preserving pooling', 'augmentation_and_loss': 'Seeded time/frequency masks and class/recording-balanced BCE with gradient accumulation', 'window_aggregation_calibration': 'Mean window probability per recording and separate highest-threshold F1 calibration'}
    assert tests['checks']=={'explicit_dft_parseval': True, 'dual_branch_gradients': True, 'mask_bounds_repeatability': True, 'recording_balanced_scaling': True, 'class_recording_balanced_loss': True, 'mean_window_aggregation': True, 'calibration_labels_do_not_change_scores': True, 'query_batch_independence': True, 'independent_f1_search': True, 'subject_disjointness': True, 'strict_json_boundary': True, 'mutated_intermediate_rejected_before_fit': True}
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 4, 'calibration_rows': 4, 'query_rows': 2, 'epochs': 40, 'training_windows': 18, 'query_windows': 7, 'training_improved': True, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('biosignal_sequence_*.py')}<=set(execution['code_sha256'])
    from sciona.biosignal_sequence_graph import build_biosignal_sequence_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.biosignal_sequence_execution as provider
    graph=build_biosignal_sequence_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'biosignal_sequence_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_biosignal_sequence_'+node.node_id)(witness)
    assert witness=={'kind':'BiosignalSequence.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/biosignal-sequence-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'filelock', 'jinja2', 'numpy', 'networkx', 'sympy', 'typing-extensions', 'fsspec', 'setuptools', 'mpmath', 'torch', 'markupsafe'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'BiosignalSequence-torch-0-LICENSE', 'BiosignalSequence-numpy-6-LICENSE', 'BiosignalSequence-numpy-15-LICENSE.md', 'BiosignalSequence-typing-extensions-0-LICENSE', 'BiosignalSequence-setuptools-3-LICENSE', 'BiosignalSequence-setuptools-9-LICENSE.APACHE', 'BiosignalSequence-numpy-0-LICENSE.txt', 'BiosignalSequence-numpy-2-COPYING', 'BiosignalSequence-numpy-5-LICENSE.md', 'BiosignalSequence-numpy-12-LICENSE.md', 'BiosignalSequence-setuptools-12-LICENSE', 'BiosignalSequence-numpy-4-dragon4_LICENSE.txt', 'BiosignalSequence-setuptools-0-LICENSE', 'BiosignalSequence-setuptools-10-LICENSE.BSD', 'BiosignalSequence-numpy-13-LICENSE.md', 'BiosignalSequence-numpy-3-LICENSE', 'BiosignalSequence-jinja2-0-LICENSE.txt', 'BiosignalSequence-numpy-10-LICENSE.md', 'BiosignalSequence-numpy-11-LICENSE.md', 'BiosignalSequence-numpy-14-LICENSE.md', 'BiosignalSequence-setuptools-4-LICENSE', 'BiosignalSequence-numpy-9-LICENSE', 'BiosignalSequence-setuptools-13-LICENSE.txt', 'BiosignalSequence-numpy-16-LICENSE.md', 'BiosignalSequence-setuptools-2-LICENSE', 'BiosignalSequence-setuptools-11-LICENSE', 'BiosignalSequence-markupsafe-0-LICENSE.txt', 'BiosignalSequence-sympy-0-LICENSE', 'BiosignalSequence-setuptools-7-LICENSE', 'BiosignalSequence-setuptools-8-LICENSE', 'BiosignalSequence-numpy-8-LICENSE.txt', 'BiosignalSequence-networkx-0-LICENSE.txt', 'BiosignalSequence-setuptools-5-LICENSE', 'BiosignalSequence-numpy-1-LICENSE.txt', 'BiosignalSequence-setuptools-6-LICENSE', 'BiosignalSequence-filelock-0-LICENSE', 'BiosignalSequence-numpy-7-LICENSE.md', 'BiosignalSequence-setuptools-1-LICENSE', 'BiosignalSequence-fsspec-0-LICENSE', 'BiosignalSequence-mpmath-0-LICENSE', 'BiosignalSequence-setuptools-14-LICENSE'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_biosignal_sequence_execution.py','requirements/biosignal-sequence-execution.txt']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='biosignal-sequence-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_biosignal_sequence_{s}.json':sha(reviews/f'competition_biosignal_sequence_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_biosignal_sequence_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
