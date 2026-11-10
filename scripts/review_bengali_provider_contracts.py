"""Bind explicit provider contracts to the complete executed Bengali graph.

This is a read-only contract review, not a license clearance or publication.
"""
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from scripts import bengali_complete_evidence as evidence

ROOT = evidence.ROOT
COMMIT = '6fb098a30429ccde1c4beb39daed250a90d1157a'
PREFIX = 'sciona.atoms.dl.grapheme_execution.'
IMAGES = ('Ordered grayscale uint8 images with shape (N,137,236). Preserve row order across all '
          'branches. Caller establishes training/calibration/query independence. This port does not '
          'validate that independence or support arbitrary image geometry.')
B7 = ('Mapping with population, targets, validation, validation_targets, branch, options, '
      'initialization_path and output_directory. Populations are validated ImagePopulation objects; '
      'targets are aligned integer class indices. Training length must divide batch size. Options '
      'contain exactly epochs, batch_size, seed, mean, std, warmup_steps, first_schedule_index and '
      'checkpoint_policy. Initialization is an explicit trusted local standard B7 checkpoint. '
      'Normalization, schedule and selected checkpoint are caller policies, not authenticated winner settings.')
FONT = ('Mapping with validated ImagePopulation, options and output_directory. Options contain '
        'exactly epochs, batch_size, seed and backend. Population uses the fixed 14784-class joint '
        'ontology; font training uses B0, source augmentation and a best validation checkpoint. '
        'Caller supplies licensed images and prevents validation leakage.')
GAN = ('Mapping with hand and font ImagePopulation objects, options and output_directory; '
       'classifier_state arrives on a separate graph edge. Options contain exactly epochs, '
       'batch_size, seed, backend and classifier_weight. Two generators and two discriminators '
       'train while the associated font classifier stays frozen. Label-preserving translation '
       'is an application assumption, not a consequence of the loss.')
STATE = ('Trusted local checkpoint state with kind, path, training-time SHA256, metrics and policy. '
         'B7 adds branch and normalization; generator adds its associated font-classifier state. '
         'Inference verifies checkpoint bytes before fresh tensor-only loading. Paths require a '
         'provisioned private filesystem; model checkpoints do not resume optimizer/RNG history.')
MASK = ('Boolean vector of length14784 identifying known joint classes, aligned with the fixed '
        'ontology. Use identical membership for calibration and query routing. Membership and '
        'unseen-class claims require caller provenance.')
SCORES = ('Aligned mapping containing ood_confidence and seen_joint_classes. Confidence is maximum '
          'sigmoid over1295 OOD logits; seen selection is float32 softmax then argmax over14784 '
          'classes. Maximum sigmoid is not calibrated uncertainty.')
UNSEEN = ('Two-font ensemble mapping including CPU joint_classes tensor. Each branch separately '
          'softmaxes14784 logits and marginalizes components before ensemble component selection. '
          'Independent component maxima can produce a joint class neither branch selected directly.')
CALIBRATION = ('Fitted finite threshold and grid scores from independent calibration rows. '
               'Weighted component macro recall uses explicit sample weights and2:1:1 component '
               'weights. Ties choose the smallest supplied threshold. Grid optimality is not '
               'unbiased application performance.')
PORTS = {
    'validation_images': IMAGES, 'query_images': IMAGES, 'images': IMAGES,
    'ood_spec': B7+' Branch must be ood; targets are in [0,1295).',
    'seen_spec': B7+' Branch must be seen; targets are in [0,14784).',
    'font_a_spec': FONT, 'font_b_spec': FONT,
    'gan_a_spec': GAN, 'gan_b_spec': GAN,
    'classifier_state': STATE+' Kind must be font.', 'state': STATE,
    'ood_logits': 'Finite CPU float32 matrix (N,1295), aligned with seen logits.',
    'seen_logits': 'Finite CPU float32 matrix (N,14784), aligned with OOD logits.',
    'logits_a': 'Finite CPU float32 matrix (N,14784), aligned with the other font branch.',
    'logits_b': 'Finite CPU float32 matrix (N,14784), aligned with the other font branch.',
    'classifier_scores': SCORES, 'unseen_predictions': UNSEEN,
    'truth_components': 'Aligned integer calibration labels (N,3) in canonical168/11/7 component ranges.',
    'known_class_mask': MASK,
    'sample_weight': 'Aligned finite nonnegative calibration weights with positive total; caller justifies target mixture.',
    'thresholds': 'Nonempty finite candidate threshold vector in [0,1]; calibration searches only this grid.',
    'calibration': CALIBRATION,
    'joint_classes': 'Aligned integer joint predictions in [0,14784); fixed168/11/8 mixed-radix ontology.',
}
OUTPUTS = {
    'validation_images': IMAGES, 'query_images': IMAGES, 'state': STATE,
    'logits': 'Finite CPU float32 logits in input row order;1295 classes for OOD,14784 for seen/font. Not probabilities.',
    'scores': SCORES, 'predictions': UNSEEN, 'calibration': CALIBRATION,
    'joint_classes': PORTS['joint_classes']+' Confidence below threshold selects alternate only when alternate is unknown; equality retains seen.',
    'components': 'Integer canonical component predictions (N,3). Decode168/11/8 then correct consonant7 to2; output has168/11/7 bounds.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    qualification = evidence.review(COMMIT)
    from sciona.grapheme_execution import build_grapheme_graph
    graph = build_grapheme_graph()
    providers = {}
    roots = {}
    consumed = {(edge.target_id, edge.input_name) for edge in graph.edges}
    produced = {(edge.source_id, edge.output_name) for edge in graph.edges}
    terminal = []
    for node in graph.nodes:
        fqdn = node.matched_primitive
        if not fqdn.startswith(PREFIX):
            raise ValueError('Unexpected provider namespace')
        function = getattr(importlib.import_module(PREFIX[:-1]), fqdn[len(PREFIX):])
        signature = inspect.signature(function)
        if list(signature.parameters) != [port.name for port in node.inputs]:
            raise ValueError('Provider input order differs: '+fqdn)
        inputs = [dict(name=p.name, type_desc=p.type_desc, required=p.required,
                       constraints=PORTS[p.name]) for p in node.inputs]
        outputs = [dict(name=p.name, type_desc=p.type_desc,
                        constraints=OUTPUTS[p.name]) for p in node.outputs]
        entry = dict(runtime_fqdn=fqdn, inputs=inputs, outputs=outputs)
        if fqdn in providers and providers[fqdn] != entry:
            raise ValueError('Repeated provider has inconsistent contracts')
        providers[fqdn] = entry
        for port in inputs:
            if (node.node_id, port['name']) not in consumed:
                if port['name'] in roots and roots[port['name']] != port:
                    raise ValueError('Shared root input contract differs')
                roots[port['name']] = port
        terminal.extend(dict(node_id=node.node_id, **port) for port in outputs
                        if (node.node_id, port['name']) not in produced)
    if (len(providers), len(roots), len(terminal)) != (14, 12, 1):
        raise ValueError('Complete provider and root boundary required')
    paths = [evidence.PROVIDER, 'sciona/grapheme_execution.py',
             'sciona/grapheme_workflow_runtime.py', 'scripts/bengali_complete_evidence.py',
             'scripts/review_bengali_provider_contracts.py', 'docs/reviews/bengali_reuse_boundaries.json']
    return dict(format='bengali-provider-contracts.v1', passed=True, approved=False,
        catalog_mutations=0, execution_qualification=qualification,
        providers=list(providers.values()), root_inputs=list(roots.values()), root_outputs=terminal,
        implementation_sha256={p: sha(ROOT/p) for p in paths},
        limitations=[
            'Read-only implementation-bound contract review. No provider/catalog approval or source-license clearance.',
            'Fixed grapheme ontology, geometry and architectures require explicit adapters for other domains. Generic routing reuse is qualified separately.',
            'Kind witnesses are symbolic summaries; they do not enforce full shape, alignment, provenance or memory contracts.',
            'Caller must provision distinct trusted output directories and bound native threads, loaders and memory across all stages.',
            'Source licensing, upstream attribution, candidate initialization permissions, corrected stored execution and transactional publication remain outstanding.',
        ])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/bengali_provider_contracts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, providers=len(result['providers']),
        root_inputs=len(result['root_inputs']), root_outputs=len(result['root_outputs']))))
