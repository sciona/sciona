"""Qualify reuse boundaries without treating recipe adapters as general algorithms."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TESTS = ['tests/test_confidence_routing.py', 'tests/test_grapheme_routing_calibration.py',
         'tests/test_grapheme_classifier_inference.py']
BOUNDARIES = [
    dict(name='confidence_routing', role='reusable_core',
        files=['sciona/confidence_routing.py'],
        contract='Aligned finite numeric vectors, confidence in [0,1], boolean alternate eligibility, common higher-is-better utility scale, nonnegative weights with positive total, explicit finite threshold grid in [0,1]. Numeric inputs become float64. Equality retains the primary route; utility ties choose the smallest threshold.',
        transfer='Constructed manufacturing classification utility and energy-dispatch negative-cost utility exercise the same implementation with an independent scalar score oracle.',
        limitation='The domain adapter supplies eligibility, utility units, validation independence and deployment mixture. Optimality covers only the supplied grid. Calibration utility is not an unbiased performance estimate; lower-precision boundary comparisons require separate qualification.'),
    dict(name='component_metric', role='reusable_core_with_domain_adapter',
        files=['sciona/component_recall.py', 'sciona/grapheme_routing_calibration.py'],
        contract='Generic aligned integer component labels and explicit sample/component weights produce additive macro-recall utilities. The grapheme adapter owns 168/11/8 decoding, consonant 7-to-2 correction, canonical 168/11/7 truth bounds and 2:1:1 metric weights.',
        transfer='Component recall can apply to other discrete multi-component outputs under an explicit ontology and weighting policy; the fixed grapheme decoder must be replaced for a different vocabulary.',
        limitation='Per-row utilities already include the metric sample weights; calibration must not apply those weights twice. Metric equivalence is tested with independent route-then-score calculations, not real application quality.'),
    dict(name='population_and_class_membership', role='domain_adapter',
        files=['sciona/bengali_population.py', 'sciona/grapheme_workflow_runtime.py', 'sciona/grapheme_b7_preprocessing.py', 'sciona/bengali_preprocessing.py'],
        contract='Validated grayscale uint8 populations and aligned joint-class labels; B7 images remain 137 by 236 and are replicated into RGB with caller-supplied channel normalization. The separate font/GAN path crops six columns from each side, resizes to 224 by 224 and uses sequential divide-by-255, subtract-0.5, divide-by-0.5 normalization. Seen membership is an explicit 14784-entry boolean mask; OOD training uses a caller mapping into 1295 classes.',
        transfer='Image-to-symbol applications need their own label ontology, class-membership provenance, image geometry and acquisition normalization. No general image modality or arbitrary class-count support is inferred.',
        limitation='Separating calibration and query graph inputs does not prove population independence. Callers must prevent training/calibration/query leakage and justify unseen-class membership. Font-only synthetic classes test routing structure, not zero-shot accuracy.'),
    dict(name='training_and_checkpoint_state', role='recipe_adapter',
        files=['sciona/grapheme_workflow_runtime.py', 'sciona/grapheme_b7_fit.py',
               'sciona/bengali_classifier_fit.py', 'sciona/bengali_fit.py'],
        contract='Separate B7 OOD/seen fits, two font-classifier fits and two guided CycleGAN fits expose model-state handoffs. Checkpoint digests are verified before fresh tensor-only restoration; frozen classifiers remain fixed while generator gradients flow through them.',
        transfer='The directed combination of classification, domain translation and confidence routing is a reusable recipe when font-like target domains, label preservation and valid confidence semantics are independently justified.',
        limitation='Current adapters have fixed architectures and class widths. Checkpoint paths require a provisioned trusted local filesystem; model-only checkpoints do not resume optimizer/RNG history. Explicit training policies and candidate pretrained initialization do not authenticate exact historical winner choices.'),
    dict(name='score_and_ensemble_semantics', role='domain_adapter',
        files=['sciona/grapheme_classifier_inference.py', 'sciona/bengali_unseen_ensemble.py', 'sciona/bengali_routing.py'],
        contract='Finite aligned CPU float32 logits use 1295 OOD and 14784 seen/unseen classes. OOD confidence is maximum sigmoid; seen selection applies softmax then joint argmax. Two font branches separately softmax and marginalize before component selection; low-confidence alternate predictions fall back to seen when alternate class membership is known.',
        transfer='Class confidence, ontology-preserving marginalization and fallback rules must be supplied and evaluated for each application. The generic confidence core does not infer these meanings.',
        limitation='Maximum sigmoid is not calibrated uncertainty. Independent component maxima may form a joint class neither model selected directly. Seen decoding is an explicit reconstruction choice, and float32 softmax tie behavior is retained.'),
    dict(name='graph_and_selection', role='composition_boundary',
        files=['sciona/grapheme_execution.py', '../sciona-atoms-dl/src/sciona/atoms/dl/grapheme_execution.py'],
        contract='Twenty-two nodes separate all six fits, calibration/query inference, score extraction, two-font ensemble, threshold fitting, routing and output conversion. Fourteen provider callables use trusted object boundaries and lightweight kind witnesses.',
        transfer='Selection must use computational applicability and declared adapter assumptions independently of competition provenance. Generic routing reuse does not qualify every recipe stage for standalone cross-disciplinary selection.',
        limitation='Object ports and kind witnesses alone do not establish shapes, population alignment, resource safety or catalog eligibility. Full execution, source/license review, complete catalog port constraints and transactional publication remain separate requirements.'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    paths = set(TESTS) | {name for item in BOUNDARIES for name in item['files']} | {str(Path(__file__).relative_to(ROOT))}
    hashes = {name: sha(ROOT/name) for name in sorted(paths)}
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1')
    with tempfile.TemporaryDirectory(prefix='synthetic-reuse-tests-') as directory:
        xml = Path(directory)/'tests.xml'
        subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS, '--junitxml='+str(xml)],
                       cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
        cases = ET.parse(xml).findall('.//testcase')
        if len(cases) != 33 or any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
            raise ValueError('All 33 synthetic reuse and adapter checks required')
    if hashes != {name: sha(ROOT/name) for name in hashes}:
        raise ValueError('Reviewed implementation changed')
    return dict(format='bengali-reuse-boundaries.v1', passed=True, approved=False, catalog_mutations=0,
        scope='Implementation-bound reuse review and synthetic core/adapter tests; not full workflow or publication qualification.',
        boundaries=BOUNDARIES, synthetic_tests_passed=len(cases), implementation_sha256=hashes,
        empirical_cross_domain_validation=False,
        reuse_decision='Expose confidence routing and component-recall computation separately from fixed grapheme ontologies and font-model adapters. Transfer of the complete recipe remains conditional on target-domain semantics, label preservation, confidence behavior and independent population evaluation.')


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/bengali_reuse_boundaries.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, synthetic_tests_passed=result['synthetic_tests_passed'])))
