"""Execute the winner's two B7 schedule snippets against the reconstruction."""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np

from sciona.grapheme_b7_schedule import warmup_linear_multiplier


BODY_SHA256 = '762fa0cf4ef4de04840a66ea83b023a29ed57ca6745e489e49f02986a9edcb07'
MARKDOWN_SHA256 = 'e7baa14c4dec0cf1d579fdaa79f5d4f8e2fabd916e14d5da40d52e085fb4f6f4'


def find_body(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'rawMarkdown' and isinstance(item, str):
                yield item
            else:
                yield from find_body(item)
    elif isinstance(value, list):
        for item in value:
            yield from find_body(item)


def main(topic, output):
    topic_data = json.loads(topic.read_text())
    content = topic_data['forumTopic']['writeUp']['message']['content']
    if hashlib.sha256(content.encode()).hexdigest() != BODY_SHA256:
        raise ValueError('pinned rendered winner post required')
    bodies = [b for b in find_body(topic_data)
              if hashlib.sha256(b.encode()).hexdigest() == MARKDOWN_SHA256]
    if len(bodies) != 1:
        raise ValueError('one pinned winner post body required')
    text = html.unescape(bodies[0]).split('## (3)')[0]
    functions = []
    for code in re.findall(r'```python=?\n(.*?)```', text, re.S):
        if 'def warmup_linear_decay(' not in code:
            continue
        for node in ast.parse(code).body:
            if isinstance(node, ast.FunctionDef) and node.name == 'warmup_linear_decay':
                functions.append(node)
    if len(functions) != 2:
        raise ValueError('both seen and OOD source schedules required')
    rng = np.random.default_rng(2063)
    checks = 0
    for function in functions:
        for index in range(128):
            total = int(rng.integers(2, 20000))
            warmup = float(rng.uniform(.1, total - .1))
            if index % 2 == 0:
                warmup = int(rng.integers(1, total))
            namespace = dict(WARM_UP_STEP=warmup, train_steps=total)
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<winner-schedule>', 'exec'), namespace)
            steps = {0, total, max(0, int(warmup)-1), int(warmup), min(total, int(warmup)+1)}
            steps.update(int(x) for x in rng.integers(0, total+1, 24))
            for step in steps:
                assert warmup_linear_multiplier(step, total_steps=total, warmup_steps=warmup) == namespace['warmup_linear_decay'](step)
                checks += 1
    root = Path(__file__).resolve().parents[1]
    files = ['sciona/grapheme_b7_schedule.py', 'scripts/validate_bengali_b7_schedule.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
                  source_body_sha256=BODY_SHA256, source_markdown_sha256=MARKDOWN_SHA256,
                  branches=2, schedule_configurations=256,
                  exact_multiplier_checks=checks, winner_warmup_duration_recovered=False,
                  implementation_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
                  limits=['Formula equality only; synthetic warmup parameters are not attributed to the winner.',
                          'Full B7 training, scheduler/optimizer ordering, normalization and warmup identity remain pending.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.topic, args.output)
