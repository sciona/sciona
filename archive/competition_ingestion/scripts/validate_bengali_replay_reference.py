"""Compare an explicitly supplied, hash-pinned public code reference on synthetic inputs."""
import argparse
import ast
import hashlib
import json
import random
from pathlib import Path

import torch

from sciona.bengali_image_pool import ImageReplayPool


SOURCE_SHA256 = '600d64d01226bd5ca879a347ca06f148193fd73c88039222cee1832c847bc716'


def main(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('reference source hash mismatch')
    # Execute only the reviewed class, never notebook setup or data-loading cells.
    text = raw.decode()
    # The source-only notebook also contains IPython shell cells. Bound the
    # reviewed class before parsing rather than executing or rewriting cells.
    parsed = ast.parse(text[text.index('class ImagePool():'):text.index('class GANLoss(')])
    classes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'ImagePool']
    if len(classes) != 1:
        raise ValueError('reference must have exactly one ImagePool class')
    namespace = dict(torch=torch, random=random.Random(153))
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<pinned-image-pool-reference>', 'exec'), namespace)
    reference = namespace['ImagePool'](50)
    actual = ImageReplayPool(seed=153)
    start = 0
    counts = [17,33,8,64,127,16]
    for count in counts:
        images = torch.arange(start,start+count,dtype=torch.float32).reshape(-1,1,1,1).expand(-1,3,2,2).clone()
        expected, result = reference.query(images), actual.query(images)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.stack(actual.images), torch.cat(reference.images), rtol=0, atol=0)
        start += count
    report = dict(passed=True, synthetic_only=True, catalog_mutations=0,
        compared_queries=len(counts), compared_images=start, pool_size=50,
        exact_outputs_and_history=True, source_only_sha256=SOURCE_SHA256,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path('sciona/bengali_image_pool.py').read_bytes()).hexdigest(),
        limits=['Isolated replay-pool equivalence; complete CycleGAN and winning pipeline remain unexecuted.',
                'Reconstruction clones caller inputs to prevent source history aliasing and uses explicit private RNG.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
