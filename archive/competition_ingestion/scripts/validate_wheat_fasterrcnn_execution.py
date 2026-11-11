"""Exercise a full source-configured detector update and synthetic inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from sciona.wheat_fasterrcnn import initialized_fasterrcnn


def main(checkpoint, output):
    torch.set_num_threads(2)
    torch.manual_seed(1287)
    model = initialized_fasterrcnn(checkpoint).train()
    frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
    head_before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=5e-4, momentum=.9, weight_decay=.0005)
    images = [torch.rand(3, 1024, 1024, generator=torch.Generator().manual_seed(1288))]
    targets = [dict(boxes=torch.tensor([[80., 90., 280., 310.], [430., 470., 810., 890.]]),
                    labels=torch.ones(2, dtype=torch.int64))]
    losses = model(images, copy.deepcopy(targets))
    assert set(losses) == {'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'}
    assert all(torch.isfinite(loss) and loss >= 0 for loss in losses.values())
    total = sum(losses.values())
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    optimizer.step()
    assert not torch.equal(head_before, model.roi_heads.box_predictor.cls_score.weight)
    for key, parameter in model.named_parameters():
        if key in frozen:
            torch.testing.assert_close(parameter, frozen[key], rtol=0, atol=0)
    print(json.dumps(dict(training_update_passed=True, losses={k: float(v.detach()) for k, v in losses.items()})), flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    assert len(predictions) == 1
    prediction = predictions[0]
    count = len(prediction['scores'])
    assert 0 < count <= 100 and prediction['boxes'].shape == (count, 4)
    assert all(torch.isfinite(value).all() for value in prediction.values())
    assert (prediction['labels'] == 1).all()
    assert (prediction['boxes'] >= 0).all() and (prediction['boxes'] <= 1024).all()
    assert ((prediction['scores'] > .05) & (prediction['scores'] <= 1)).all()
    files = ['sciona/wheat_fasterrcnn.py', 'sciona/wheat_resnet_backbone.py', 'sciona/wheat_legacy_weights.py',
             'sciona/wheat_frozen_normalization.py', 'sciona/wheat_feature_pyramid.py', 'sciona/wheat_proposals.py',
             'sciona/wheat_anchors.py', 'sciona/wheat_detection_losses.py', 'sciona/wheat_image_transform.py',
             'scripts/validate_wheat_fasterrcnn_execution.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        input_shape=[1, 3, 1024, 1024], internal_image_size=800, training_updates=1,
        losses={k: float(v.detach()) for k, v in losses.items()},
        all_trainable_parameter_gradients_finite=True, trainable_parameter_tensors=len(parameters),
        frozen_parameters_unchanged=True, predictor_updated=True, inference_detections=count,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['One synthetic SGD step and inference; not a complete source training fit or accuracy claim.',
                'Individual components have separate historical comparisons; full assembled historical-reference equivalence remains to be checked.',
                'Installed native kernels; full base-fit inventory, ensemble and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.output)
