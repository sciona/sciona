"""Actual source-mechanics updates starting from reviewed reference backbones."""
import gc
import hashlib
import json
from pathlib import Path

import torch

from sciona.cassava_resnext import Classifier as ResNeXt, optimizer_for, train_epoch
from sciona.cassava_vit_model import Classifier as ViT
from sciona.cassava_vit_epoch import train_epoch as vit_epoch
from sciona.cassava_vit_ordered_loss import loss


def main():
    torch.set_num_threads(2)
    evidence = json.loads(Path('docs/reviews/competition_cassava_timm_pretrained_validation.json').read_text())
    assert evidence['passed'] and not evidence['approved']
    for path, expected in evidence['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    results = {}
    for family, classifier, size, tag, parameter in [
        ('resnext', ResNeXt, 512, 'resnext50_32x4d.ra_in1k', 'conv1.weight'),
        ('vit', ViT, 384, 'vit_base_patch16_384.orig_in21k_ft_in1k', 'patch_embed.proj.weight'),
    ]:
        artifact = evidence['references'][tag]
        torch.manual_seed(971)
        model = classifier(pretrained=True,
            weights_path=Path('/private/tmp/sciona_cassava_timm_pretrained') / f'{tag}.safetensors',
            expected_sha256=artifact['weights_sha256'])
        before = model.model.state_dict()[parameter].clone()
        head = model.model.fc if family == 'resnext' else model.model.head
        assert head.out_features == 5
        head_before = head.weight.detach().clone()
        batches = [(torch.randn(1, 3, size, size), torch.tensor([label])) for label in (1, 3)]
        if family == 'resnext':
            optimizer = optimizer_for(model)
            value = train_epoch(model, batches, optimizer)
            assert torch.isfinite(torch.tensor(value))
            updates = 2
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 / 7)
            result = vit_epoch(model, batches, optimizer, loss_function=loss)
            updates = result['optimizer_steps']
            assert updates == 1
        assert not torch.equal(before, model.model.state_dict()[parameter])
        assert not torch.equal(head_before, head.weight)
        for value in model.state_dict().values():
            assert torch.isfinite(value).all()
        model.eval()
        with torch.inference_mode():
            probabilities = model(batches[0][0]).softmax(-1)
        assert probabilities.shape == (1, 5) and torch.isfinite(probabilities).all()
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
        results[family] = {'pretrained_sha256': artifact['weights_sha256'],
            'full_resolution_microbatches': 2, 'optimizer_steps': updates,
            'backbone_and_five_class_head_updated': True, 'all_state_finite': True,
            'post_update_five_class_probabilities_valid': True}
        del model, optimizer, batches, head, before, head_before
        gc.collect()
        print(f'{family}: pretrained source-mechanics update passed', flush=True)
    paths = ['sciona/cassava_pretrained.py', 'sciona/cassava_resnext.py',
        'sciona/cassava_vit_model.py', 'sciona/cassava_vit_epoch.py',
        'sciona/cassava_vit_ordered_loss.py', 'sciona/cassava_loss.py',
        'scripts/validate_cassava_pretrained_training.py',
        'docs/reviews/competition_cassava_timm_pretrained_validation.json']
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
        'families': results,
        'scope': 'Pretrained reconstruction initialization and individual source-mechanics updates; full pretrained folds, original execution environment and ensemble remain unqualified.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_pretrained_training_validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
