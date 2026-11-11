"""Compare candidate components against pinned public code on synthetic inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_competition_dsb_semantics import checked_source, source_oversampling
from sciona.dsb_components import combine_volume, oversample_by_size, sample_proposals, split_volume, process_lung_mask, mask_and_remove_bone
from sciona.dsb_components import preprocess_from_masks
from sciona.dsb_components import segment_volume
from sciona.dsb_components import assign_anchor_labels
from sciona.dsb_components import decode_proposals, suppress_proposals
from sciona.dsb_components import prepare_detector_tiles
from sciona.dsb_components import crop_classifier_proposal
from sciona.dsb_components import prepare_classifier_batch


def validate(root, source_root):
    pins = json.loads((root / 'docs/reviews/competition_dsb_source_pins.json').read_text())
    source = checked_source(source_root, pins, 'split_combine.py')
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    # Only the two audited integer augmented divisions need Python 3 adaptation.
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            if not isinstance(node.target, ast.Name) or node.target.id not in ['side_len', 'margin']:
                raise ValueError('Unexpected source division')
            node.op = ast.FloorDiv()
            changes += 1
    if changes != 2:
        raise ValueError('Source division inventory changed')
    scope = dict(np=np)
    exec(compile(tree, '<pinned-split-combine-python3>', 'exec'), scope)
    rng = np.random.RandomState(17)
    volume_cases = 0
    for shape in [(1, 1, 3, 5), (2, 7, 9, 11), (1, 8, 8, 8)]:
        for stride in [1, 2]:
            for margin in [0, 2]:
                volume = rng.normal(size=shape)
                reference = scope['SplitComb'](4, 2, stride, margin, 170)
                expected, grid = reference.split(volume)
                actual, actual_grid = split_volume(volume, 4, 2, margin)
                np.testing.assert_array_equal(actual, expected)
                assert tuple(grid) == actual_grid
                extent = (4 + 2 * margin) // stride
                predictions = rng.normal(size=(len(actual), extent, extent, extent, 3, 5))
                expected = reference.combine(predictions, nzhw=grid)
                actual = combine_volume(predictions, actual_grid, 4, stride, margin)
                np.testing.assert_array_equal(actual, expected)
                assert actual.dtype == expected.dtype == np.float32
                volume_cases += 1

    sampling_source = checked_source(source_root, pins, 'data_classifier.py')
    tree = ast.parse(sampling_source[sampling_source.index('def sample('):])
    # Python 2 range returns a mutable list used by the author sampler.
    scope = dict(np=np, range=lambda *args: list(range(*args)))
    exec(compile(tree, '<pinned-sampler-python3>', 'exec'), scope)
    sampling_cases = 0
    for scores in [np.array([0., -1000., -2000.]), rng.normal(size=9), np.zeros(5)]:
        for k in [0, 1, len(scores)-1, len(scores), len(scores)+1]:
            for seed in [0, 7, 91]:
                np.random.seed(seed)
                expected = scope['sample'](scores.copy(), k, .7)
                actual = sample_proposals(scores, k, .7, seed)
                np.testing.assert_array_equal(actual, expected)
                sampling_cases += 1
    detector_source = checked_source(source_root, pins, 'data_detector.py')
    labels = rng.normal(size=(40, 4))
    labels[:, 3] = np.r_[5., 6., 7., 30., 31., 40., 41., rng.uniform(1, 70, 33)]
    expected = source_oversampling(detector_source, [labels])[:, 1:]
    np.testing.assert_array_equal(oversample_by_size(labels, 3), expected)
    import textwrap
    from scipy.ndimage import binary_dilation, generate_binary_structure
    from skimage.morphology import convex_hull_image
    preprocessing = checked_source(source_root, pins, 'preprocessing/full_prep.py')
    tree = ast.parse(preprocessing)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ['process_mask', 'lumTrans']]
    if len(tree.body) != 2:
        raise ValueError('Unexpected preprocessing function coverage')
    scope = dict(np=np, binary_dilation=binary_dilation,
                 generate_binary_structure=generate_binary_structure, convex_hull_image=convex_hull_image)
    exec(compile(tree, '<pinned-mask-functions>', 'exec'), scope)
    start = preprocessing.index('        convex_mask = m1')
    stop = preprocessing.index('        sliceim1,_ = resample', start)
    block = compile(textwrap.dedent(preprocessing[start:stop]), '<pinned-bone-replacement>', 'exec')
    preprocessing_cases = 0
    for kind in ['empty', 'compact', 'separated', 'overlapping']:
        left = np.zeros((3, 31, 33), dtype=bool)
        right = np.zeros_like(left)
        if kind == 'compact':
            left[:, 12:17, 12:17] = True
            left[:, 14, 14] = False
        elif kind == 'separated':
            left[:, 2, 2] = True
            left[:, 28, 29] = True  # Hull guard must reject large expansion.
        elif kind == 'overlapping':
            left[:, 10:15, 10:15] = True
            right[:, 12:17, 12:17] = True
        for dtype in [np.float32, np.float64, np.int16]:
            volume = rng.uniform(-1600, 1000, left.shape).astype(dtype)
            if dtype != np.int16:
                volume[0, 0, :3] = [np.nan, np.inf, -np.inf]
            scope.update(m1=left.copy(), m2=right.copy(), im=volume.copy())
            exec(block, scope)
            np.testing.assert_array_equal(process_lung_mask(left), scope['process_mask'](left))
            np.testing.assert_array_equal(mask_and_remove_bone(volume, left, right), scope['sliceim'])
            preprocessing_cases += 1
    import warnings
    from scipy.ndimage import zoom
    tree = ast.parse(preprocessing)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'resample']
    scope.update(warnings=warnings, zoom=zoom)
    exec(compile(tree, '<pinned-source-resampling>', 'exec'), scope)
    start = preprocessing.index('        newshape = np.round')
    stop = preprocessing.index('        convex_mask = m1', start)
    crop_bounds = compile(textwrap.dedent(preprocessing[start:stop]), '<pinned-source-crop-bounds>', 'exec')
    composed_cases = 0
    for spacing in [np.array([1., 1., 1.]), np.array([1.3, .7, 2.1]), np.array([.6, 1.5, .9])]:
        for edge in [False, True]:
            volume = rng.uniform(-1500, 900, (13, 15, 17))
            left = np.zeros(volume.shape, dtype=bool)
            left[0:2, 0:2, 0:2] = edge
            left[6:8, 7:9, 8:10] = True
            right = np.zeros_like(left)
            scope.update(Mask=left, spacing=spacing, resolution=np.ones(3), m1=left, m2=right, im=volume.copy())
            exec(crop_bounds, scope)
            exec(block, scope)
            expected_full, effective = scope['resample'](scope['sliceim'], spacing, np.ones(3), order=1)
            bounds = scope['extendbox']
            expected = expected_full[tuple(slice(lo, hi) for lo, hi in bounds)]
            actual, actual_spacing, actual_bounds = preprocess_from_masks(volume, left, right, spacing)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_bounds, bounds)
            np.testing.assert_array_equal(actual_spacing, effective)
            composed_cases += 1
    segmentation = checked_source(source_root, pins, 'preprocessing/step1.py')
    tree = ast.parse(segmentation)
    names = {'binarize_per_slice', 'all_slice_analysis', 'fill_hole', 'two_lung_only', 'step1_python'}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    halves = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == 'in1d':
            node.attr = 'isin'
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and node.right.value == 2:
            node.op = ast.FloorDiv()
            halves += 1
    if halves != 8:
        raise ValueError('Unexpected segmentation integer-division inventory')
    import scipy.ndimage
    from skimage import measure
    segmentation_scope = dict(np=np, scipy=scipy, measure=measure,
                              load_scan=lambda value: value,
                              get_pixels_hu=lambda value: value)
    exec(compile(tree, '<pinned-array-segmentation>', 'exec'), segmentation_scope)
    segmentation_cases = 0
    for size in [64, 65]:
        z, y, x = np.indices((24, size, size))
        first = ((z-12)/9)**2 + ((y-32)/17)**2 + ((x-20)/9)**2 < 1
        second = ((z-12)/9)**2 + ((y-32)/17)**2 + ((x-44)/9)**2 < 1
        for count in [0, 1, 2]:
            volume = np.full(z.shape, 50.)
            if count: volume[first] = -900.
            if count == 2: volume[second] = -900.
            spacing = np.array([6., 6., 6.])
            _, expected_first, expected_second, _ = segmentation_scope['step1_python']((volume.copy(), spacing))
            actual_first, actual_second = segment_volume(volume, spacing)
            np.testing.assert_array_equal(actual_first, expected_first)
            np.testing.assert_array_equal(actual_second, expected_second)
            if count == 2 and (not actual_first.any() or not actual_second.any()):
                raise ValueError('Two-component parity case was vacuous')
            segmentation_cases += 1
    import random
    start = detector_source.index('class LabelMapping')
    stop = detector_source.index('def collate', start)
    tree = ast.parse(detector_source[start:stop])
    divisions = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.left, ast.Subscript) and isinstance(node.left.value, ast.Name)
                and node.left.value.id == 'input_size'):
            node.op = ast.FloorDiv()
            divisions += 1
    assert divisions == 1
    anchor_scope = dict(np=np, random=random.Random(), binary_dilation=binary_dilation,
                        generate_binary_structure=generate_binary_structure)
    exec(compile(tree, '<pinned-label-mapping-python3>', 'exec'), anchor_scope)
    anchor_cases = 0
    targets = [None, np.array([17.5, 17.5, 17.5, 10.]), np.array([2., 2., 2., 1.])]
    for phase in ['train', 'val']:
        for seed in [0, 4, 91]:
            for target in targets:
                for negatives in [0, 7]:
                    boxes = np.array([[5.5, 5.5, 5.5, 10.], [17.5, 17.5, 17.5, 10.]])
                    config = dict(stride=4, num_neg=negatives, th_neg=.02, anchors=[10., 30., 60.],
                                  th_pos_train=.5, th_pos_val=1.)
                    anchor_scope['random'].seed(seed)
                    expected = anchor_scope['LabelMapping'](config, phase)((32, 32, 32),
                               np.full(4, np.nan) if target is None else target, boxes)
                    actual = assign_anchor_labels((32, 32, 32), target, boxes, phase=phase,
                                                  num_neg=negatives, random_state=seed)
                    np.testing.assert_array_equal(actual, expected)
                    anchor_cases += 1
    layers = ast.parse(checked_source(source_root, pins, 'layers.py'))
    selected = [node for node in layers.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ['GetPBB', 'nms', 'iou']]
    assert len(selected) == 3
    proposal_scope = dict(np=np)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-proposal-decoding>', 'exec'), proposal_scope)
    proposal_cases = 0
    for dtype in [np.float32, np.float64]:
        for shape in [(1, 1, 1, 3, 5), (3, 4, 2, 3, 5)]:
            for threshold in [-3., 0., 10.]:
                output = rng.normal(size=shape).astype(dtype)
                original, original_indices = proposal_scope['GetPBB']({'stride':4,'anchors':[10.,30.,60.]})(output, thresh=threshold, ismask=True)
                actual, actual_indices = decode_proposals(output, threshold)
                np.testing.assert_array_equal(actual, original)
                for a, b in zip(actual_indices, original_indices): np.testing.assert_array_equal(a, b)
                for overlap_threshold in [0., .05, 1.]:
                    np.testing.assert_array_equal(suppress_proposals(actual, overlap_threshold), proposal_scope['nms'](original, overlap_threshold))
                proposal_cases += 1
    from types import SimpleNamespace
    start = detector_source.index('            nz, nh, nw = imgs.shape[1:]')
    stop = detector_source.index('            return torch.from_numpy(imgs.astype', start)
    tile_tree = ast.parse(textwrap.dedent(detector_source[start:stop]))
    for node in ast.walk(tile_tree):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Attribute) and node.right.attr == 'stride'
                and not isinstance(node.left, ast.Call)):
            node.op = ast.FloorDiv()
    tile_block = compile(tile_tree, '<pinned-inference-coordinate-grid>', 'exec')
    split_tree = ast.parse(checked_source(source_root, pins, 'split_combine.py'))
    split_tree.body = [node for node in split_tree.body if isinstance(node, ast.ClassDef)]
    tile_scope = dict(np=np)
    exec(compile(split_tree, '<pinned-source-split>', 'exec'), tile_scope)
    coordinate_cases = 0
    for shape in [(1, 1, 1, 1), (1, 19, 17, 21), (2, 32, 16, 24)]:
        for margin in [0, 8]:
            volume = rng.randint(0, 256, size=shape).astype(np.uint8)
            tile_scope.update(imgs=volume.copy(), self=SimpleNamespace(stride=4, pad_value=170,
                split_comber=tile_scope['SplitComb'](16, 8, 4, margin, 170)))
            exec(tile_block, tile_scope)
            actual, coordinates, grid = prepare_detector_tiles(volume, 16, 8, 4, margin)
            np.testing.assert_array_equal(actual, tile_scope['imgs'])
            np.testing.assert_array_equal(coordinates, tile_scope['coord2'])
            assert grid == tuple(tile_scope['nzhw'])
            coordinate_cases += 1
    crop_tree = ast.parse(checked_source(source_root, pins, 'data_classifier.py'))
    crop_tree.body = [n for n in crop_tree.body if isinstance(n, ast.ClassDef) and n.name == 'simpleCrop']
    crop_divisions = 0
    for node in ast.walk(crop_tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if ((isinstance(node.left, ast.Name) and node.left.id == 'crop_size')
                    or (isinstance(node.right, ast.Attribute) and node.right.attr == 'stride')):
                node.op = ast.FloorDiv()
                crop_divisions += 1
    assert crop_divisions == 4
    crop_scope = dict(np=np, zoom=zoom, warnings=warnings)
    exec(compile(crop_tree, '<pinned-classifier-crop-python3>', 'exec'), crop_scope)
    crop_cases = 0
    state = np.random.get_state()
    try:
        for phase in ['train', 'val', 'test']:
            for scaling in [False, True]:
                for seed in [0, 9, 41]:
                    for target in [np.array([0., 2., 4., 10.]), np.array([15., 15., 15., 10.])]:
                        volume = rng.randint(0, 256, size=(1, 25, 25, 25)).astype(np.uint8)
                        config = dict(crop_size=[16,16,16], scaleLim=[.85,1.15], radiusLim=[6,100],
                                      jitter_range=.15, augtype={'scale':scaling}, stride=4, filling_value=160)
                        np.random.seed(seed)
                        original = crop_scope['simpleCrop'](config, phase)(volume, target)
                        actual = crop_classifier_proposal(volume, target, crop_size=16, phase=phase,
                                                          scale_enabled=scaling, random_state=seed)
                        assert original[0].shape == actual[0].shape == (1,16,16,16)
                        for a, b in zip(actual, original): np.testing.assert_array_equal(a, b)
                        crop_cases += 1
    finally:
        np.random.set_state(state)
    source_classifier = checked_source(source_root, pins, 'data_classifier.py')
    start = source_classifier.index('        croplist = np.zeros')
    stop = source_classifier.index("        if self.phase!='test':", start)
    batch_tree = ast.parse(textwrap.dedent(source_classifier[start:stop]))
    for node in ast.walk(batch_tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Attribute) and node.right.attr == 'stride':
                node.op = ast.FloorDiv()
    batch_block = compile(batch_tree, '<pinned-classifier-batch>', 'exec')
    classifier_batch_cases = 0
    for count in [0, 1, 7]:
        volume = rng.randint(0, 256, (1,25,25,25)).astype(np.uint8)
        proposals = np.zeros((count,5), dtype=float)
        if count:
            proposals[:,0] = np.arange(count)
            proposals[:,1:4] = rng.uniform(0,24,(count,3))
            proposals[:,4] = 10
        config = dict(crop_size=[16,16,16], scaleLim=[.85,1.15], radiusLim=[6,100],
                      jitter_range=.15, augtype={'scale':False}, stride=4, filling_value=160)
        source_self = SimpleNamespace(topk=5, crop_size=[16,16,16], stride=4, phase='test',
                                      crop=crop_scope['simpleCrop'](config,'test'))
        chosen = proposals[:,0].argsort()[::-1][:5]
        scope = dict(np=np, self=source_self, topk=5, chosenid=chosen, img=volume,
                     pbb=proposals, pbb_label=np.zeros(count))
        exec(batch_block, scope)
        actual, coords, indices = prepare_classifier_batch(volume, proposals, 5, 16)
        np.testing.assert_array_equal(actual, scope['croplist'])
        np.testing.assert_array_equal(coords, scope['coordlist'])
        np.testing.assert_array_equal(indices, chosen)
        classifier_batch_cases += 1
    paths = ['sciona/dsb_components.py', 'sciona/dsb_segmentation.py', 'sciona/dsb_anchor_mapping.py', 'scripts/validate_dsb_components.py', 'tests/test_dsb_components.py']
    return dict(approved=False, synthetic_only=True, source_commit=pins['commit'],
                volume_parity_cases=volume_cases, sampling_parity_cases=sampling_cases,
                oversampling_source_loop_parity=True, exact_array_parity=True,
                preprocessing_parity_cases=preprocessing_cases,
                composed_preprocessing_parity_cases=composed_cases,
                segmentation_parity_cases=segmentation_cases,
                anchor_assignment_parity_cases=anchor_cases,
                proposal_decode_parity_cases=proposal_cases,
                proposal_nms_parity_cases=proposal_cases * 3,
                detector_coordinate_tile_parity_cases=coordinate_cases,
                classifier_crop_parity_cases=crop_cases,
                classifier_batch_parity_cases=classifier_batch_cases,
                source_adaptations=['Two integer augmented divisions use floor division.',
                                    'Python 2 mutable range becomes list(range).',
                                    'Reference reconstruction uses explicit grid metadata.',
                                    'Eight segmentation size halves preserve Python 2 floor division; np.in1d becomes np.isin.',
                                    'Segmentation file readers are replaced with synthetic array passthrough stubs.',
                                    'Anchor grid uses integer division; random.sample reference uses a seeded local Python RNG.',
                                    'Seed parity is under current Python, not historical Python 2 sampling behavior.',
                                    'Classifier crop preserves four Python 2 integer quotients.',
                                    'Defensive scale-output trim uses target size; nominal source floor(N/s)*s cannot overshoot N.'],
                implementation_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, args.source_root)
    (root / 'docs/reviews/competition_dsb_component_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
