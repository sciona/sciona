"""Pin and execute explicit historical-reference weights, never timm defaults.

Downloads public model software only. This does not establish the winning
notebooks' package version or equivalence to their private trained checkpoints.
"""
import hashlib
import json
from pathlib import Path

import requests
import safetensors
import timm
import torch

from sciona.cassava_pretrained import load_reference_backbone


REFERENCES = {
    'resnext50_32x4d.ra_in1k': ('5f3db49db76b4ccb53aa11ad65f41647e8063426', 512),
    'vit_base_patch16_384.orig_in21k_ft_in1k': ('b3a735915218fc1ae72bca2ae77189bd40bbad2c', 384),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def get_json(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    torch.set_num_threads(2)
    root = Path('/private/tmp/sciona_cassava_timm_pretrained')
    root.mkdir(exist_ok=True)
    results = {}
    for tag, (revision, size) in REFERENCES.items():
        api = f'https://huggingface.co/api/models/timm/{tag}'
        metadata = get_json(f'{api}/revision/{revision}')
        assert metadata['sha'] == revision
        assert metadata['cardData']['license'] == 'apache-2.0'
        tree = get_json(f'{api}/tree/{revision}')
        entry = next(item for item in tree if item['path'] == 'model.safetensors')
        expected = entry['lfs']['oid']
        path = root / f'{tag}.safetensors'
        url = f'https://huggingface.co/timm/{tag}/resolve/{revision}/model.safetensors'
        if not path.exists() or digest(path) != expected:
            partial = path.with_suffix('.partial')
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            assert partial.stat().st_size == entry['size']
            assert digest(partial) == expected
            partial.replace(path)
        assert path.stat().st_size == entry['size'] and digest(path) == expected
        print(f'{tag}: verified public weights', flush=True)
        family = 'resnext' if tag.startswith('resnext') else 'vit'
        model = load_reference_backbone(family, path, expected_sha256=expected)
        model.eval()
        generator = torch.Generator().manual_seed(723)
        synthetic = torch.rand((1, 3, size, size), generator=generator)
        with torch.inference_mode():
            output = model(synthetic)
            repeated = model(synthetic)
        assert output.shape == (1, 1000) and torch.isfinite(output).all()
        torch.testing.assert_close(output, repeated, rtol=0, atol=0)
        results[tag] = {
            'revision': revision, 'publisher_license': 'apache-2.0',
            'metadata_url': f'{api}/revision/{revision}',
            'weights_url': url, 'weights_sha256': expected,
            'weights_bytes': entry['size'], 'strict_state_load': True,
            'synthetic_inference_shape': list(output.shape),
            'repeat_outputs_identical': True,
            'explicit_config_url': model.pretrained_cfg['url'],
            'current_untagged_default_url': timm.get_pretrained_cfg(tag.split('.')[0]).url,
        }
        del model, output, repeated
        print(f'{tag}: strict load and full-resolution inference passed', flush=True)
    report = {
        'approved': False, 'passed': True, 'synthetic_only': True,
        'scope': 'Publisher-pinned historical-reference reconstruction candidates; original notebook timm version and winner checkpoint identity unverified. This report covers backbone loading and inference; full pretrained ensemble training remains unqualified.',
        'runtime': {'timm': timm.__version__, 'torch': torch.__version__, 'safetensors': safetensors.__version__},
        'references': results,
        'sha256': {p: digest(Path(p)) for p in [
            'scripts/validate_cassava_timm_pretrained.py', 'sciona/cassava_pretrained.py',
            'tests/test_cassava_pretrained.py', 'docs/reviews/competition_cassava_timm_history.json',
            'docs/reviews/competition_cassava_timm_pretrained_license.json']},
    }
    Path('docs/reviews/competition_cassava_timm_pretrained_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Both reference artifacts passed; CDG remains unpublished.', flush=True)


if __name__ == '__main__':
    main()
