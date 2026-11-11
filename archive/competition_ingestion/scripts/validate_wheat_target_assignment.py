"""Compare RPN and ROI target assignment with pinned nonempty source contracts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops

from sciona.wheat_proposals import WheatRegionProposalNetwork


def main(root, output):
    definitions, sources = {}, {}
    for file, name, method, digest in [
        ('rpn.py', 'RegionProposalNetwork', 'assign_targets_to_anchors', '62ea420ac8bba44d91d3b27f46c627fcc6b91c15ae706142fcbe079c10a31751'),
        ('roi_heads.py', 'RoIHeads', 'assign_targets_to_proposals', 'dfe9ac02abbbd2b6db522d66d00210ad0d7dac1cc66cf1f0c3e9bd628f37be2c')]:
        raw = (root / 'models/detection' / file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('historical target assignment source drift')
        sources[file] = digest
        cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == name)
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
        namespace = dict(torch=torch, box_ops=box_ops)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical-targets>', 'exec'), namespace)
        definitions[file] = namespace[method]
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(1127)
    rpn = WheatRegionProposalNetwork()
    roi = SimpleNamespace(proposal_matcher=Matcher(.5, .5, False), box_similarity=box_ops.box_iou)
    cases = 0
    for seed in range(64):
        def boxes(count):
            starts = torch.rand(count, 2, generator=generator) * 700
            return torch.cat([starts, starts + torch.rand(count, 2, generator=generator) * 150 + 1], dim=1)
        targets = [dict(boxes=boxes(n), labels=torch.ones(n, dtype=torch.int64)) for n in (1, 7, 13)]
        anchors = [torch.cat([boxes(301), target['boxes'], target['boxes']]) for target in targets]
        observed = rpn.assign_targets_to_anchors(anchors, targets)
        expected = definitions['rpn.py'](rpn, anchors, targets)
        for a_group, b_group in zip(observed, expected):
            for a, b in zip(a_group, b_group):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        gt_boxes, gt_labels = [t['boxes'] for t in targets], [t['labels'] for t in targets]
        observed = RoIHeads.assign_targets_to_proposals(roi, anchors, gt_boxes, gt_labels)
        expected = definitions['roi_heads.py'](roi, anchors, gt_boxes, gt_labels)
        for a_group, b_group in zip(observed, expected):
            for a, b in zip(a_group, b_group):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        cases += 2
    # Keep the historical rejection for training images with no ground truth.
    errors = []
    for call in (rpn.assign_targets_to_anchors, lambda a, t: definitions['rpn.py'](rpn, a, t)):
        try:
            call([boxes(3)], [dict(boxes=torch.empty(0, 4))])
        except ValueError:
            errors.append(True)
    assert errors == [True, True]
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=sources, exact_assignment_cases=cases, images_per_case=3,
        exact_ground_truth_and_duplicate_anchor_matches_included=True,
        historical_rpn_empty_ground_truth_rejection_preserved=True,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Nonempty ROI ground truth only; installed ROI empty handling differs and requires a source guard at integration.',
                'Both paths use installed box IoU and qualified matcher; complete detector execution remains pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.output)
