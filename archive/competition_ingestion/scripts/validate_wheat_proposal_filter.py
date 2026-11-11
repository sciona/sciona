"""Compare complete proposal filtering with source, including saturated logits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torchvision.ops import boxes as box_ops

from sciona.wheat_proposals import WheatRegionProposalNetwork


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != '62ea420ac8bba44d91d3b27f46c627fcc6b91c15ae706142fcbe079c10a31751':
        raise ValueError('historical RPN source drift')
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'RegionProposalNetwork')
    names = {'pre_nms_top_n', 'post_nms_top_n', '_get_top_n_idx', 'filter_proposals'}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    node = ast.ClassDef(name='ReferenceFilter', bases=[], keywords=[], body=nodes, decorator_list=[])
    namespace = dict(torch=torch, torchvision=torchvision, box_ops=box_ops)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<historical-proposals>', 'exec'), namespace)
    torch.set_num_threads(2)
    actual = WheatRegionProposalNetwork()
    reference = namespace['ReferenceFilter']()
    source_settings = dict(_pre_nms_top_n={'training': 2000, 'testing': 1000},
                           _post_nms_top_n={'training': 2000, 'testing': 1000},
                           nms_thresh=.7, min_size=1e-3)
    for key, value in source_settings.items():
        assert getattr(actual, key) == value
        setattr(reference, key, value)
    generator = torch.Generator().manual_seed(1013)
    counts = [2500, 1600, 800, 400, 200]
    cases = 0
    for training in (True, False):
        actual.train(training)
        reference.training = training
        for seed in range(8):
            starts = torch.rand(2, sum(counts), 2, generator=generator) * 1100 - 100
            sizes = torch.rand(2, sum(counts), 2, generator=generator) * 250
            boxes = torch.cat([starts, starts + sizes], dim=-1)
            boxes[:, :30, 2:] = boxes[:, :30, :2]  # Removed zero-area proposals.
            scores = torch.randn(2 * sum(counts), generator=generator)
            if seed == 0:
                scores = scores + 100  # Sigmoid would erase score distinctions.
            if seed == 1:
                scores.zero_()  # Exact ties retain the source/runtime ordering.
            scores.requires_grad_()
            shapes = [(800, 800), (768, 1024)]
            observed = actual.filter_proposals(boxes, scores, shapes, counts)
            expected = reference.filter_proposals(boxes, scores, shapes, counts)
            for group_a, group_b in zip(observed, expected):
                for a, b in zip(group_a, group_b):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    assert torch.isfinite(a).all() and not a.requires_grad
            assert all(len(value) <= (2000 if training else 1000) for value in observed[0])
            cases += 1
    files = ['sciona/wheat_proposals.py', 'scripts/validate_wheat_proposal_filter.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=digest, exact_filter_cases=cases, images_per_case=2, anchors_per_image=sum(counts),
        training_and_evaluation_budgets_exercised=True, raw_logits_preserved=True,
        clipping_small_box_removal_saturation_and_ties_included=True,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Filtering only; full RPN forward, target assignment and box coding remain separate.',
                'Both paths call installed box/NMS operators; historical native-kernel equivalence is not claimed.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
