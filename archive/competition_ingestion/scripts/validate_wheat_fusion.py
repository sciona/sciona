"""Exact synthetic comparison with pinned notebook WBF wrapper and wheel."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciona.wheat_fusion import fuse_predictions, STAGES


def main(notebook, wheel_root, manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    if manifest['wheel_sha256'] != '2f39ae15304b8b60524d41b5b4ce4bdef3a5732ded5668544bf016ce0bfe67e8':
        raise ValueError('reference publisher wheel differs')
    for name, digest in manifest['files_sha256'].items():
        if hashlib.sha256((wheel_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('reference wheel payload differs')
    import ensemble_boxes
    if Path(ensemble_boxes.__file__).resolve().parent != (wheel_root / 'ensemble_boxes').resolve():
        raise ValueError('unexpected fusion package loaded')
    raw = notebook.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '3ffd2f6dbc30822d6aeb104dd1ea6bae5856ed3f2c01a639f7fee27867d111ab':
        raise ValueError('notebook reference differs')
    text = raw.decode()
    fragment = text[text.index('def run_wbf('):text.index('def run_wbf_4preds(')]
    namespace = dict(np=np, weighted_boxes_fusion=ensemble_boxes.weighted_boxes_fusion)
    exec(compile(ast.parse(fragment), '<pinned-wheat-fusion>', 'exec'), namespace)
    rng = np.random.default_rng(513)
    cases = 0
    for stage, (views, box_threshold, post_threshold) in STAGES.items():
        for case in range(24):
            boxes, scores = [], []
            for view in range(views):
                if case == 0 or (case == 1 and view > 0) or rng.random() < .15:
                    boxes.append([]); scores.append([])
                else:
                    # Repeated overlapping boxes exercise clustering across views.
                    jitter = rng.uniform(-.01, .01, (2, 4))
                    b = (np.array([[.1, .1, .4, .4], [.6, .6, .95, .95]]) + jitter).clip(0, 1)
                    boxes.append(b.tolist())
                    scores.append(rng.uniform(.3, .99, 2).tolist())
            labels = [[0] * len(b) for b in boxes]
            identity = 'synthetic_query'
            test_frame = SimpleNamespace(image_id=SimpleNamespace(values=np.array([identity])))
            expected = namespace['run_wbf'](test_frame, {identity: boxes}, {identity: scores},
                {identity: labels}, {identity: (137, 236)}, .5, box_threshold, post_threshold)[identity]
            actual = fuse_predictions(boxes, scores, stage=stage, height=137, width=236)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            cases += 1
    # Removing empty views would retain this high-confidence isolated box.
    boxes = [[[.1, .1, .4, .4]]] + [[] for _ in range(7)]
    scores = [[.99]] + [[] for _ in range(7)]
    b, s = fuse_predictions(boxes, scores, stage='pseudo1', height=137, width=236)
    assert b.shape == (0, 4) and s.shape == (0,)
    try:
        fuse_predictions(boxes[:1], scores[:1], stage='pseudo1', height=137, width=236)
    except ValueError:
        pass
    else:
        raise AssertionError('missing view inventory accepted')
    files = ['sciona/wheat_fusion.py', 'scripts/validate_wheat_fusion.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        exact_notebook_comparison_cases=cases, all_three_stage_thresholds_checked=True,
        empty_views_retained_in_confidence_denominator=True, incomplete_view_inventory_rejected=True,
        publisher_wheel_sha256=manifest['wheel_sha256'], verified_wheel_payload_files=len(manifest['files_sha256']),
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Both wrapper paths use the same hash-verified1.0.4fusion package; this is not independent WBF algorithm reimplementation.',
                'Installed NumPy/Numba backend is used; historical numeric/runtime dependency qualification remains separate.',
                'Full detector predictions, TTA reversal, pseudo training and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'implementation_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--wheel-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.notebook, args.wheel_root, args.manifest, args.output)
