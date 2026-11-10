"""Compare the serial cleaning adapter with source in a pinned current runtime."""
import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import random

from sciona.mercari_text_cleaning import clean_source_text

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '076683d8877644cf7b8f13b125e57ed596bd2a5455f531bb8f162b4df285c9bc'


def validate(source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Pinned text source required')
    dependencies = {}
    for name, expected in [('Unidecode', '1.4.0'), ('nltk', '3.9.4')]:
        package = distribution(name)
        if package.version != expected:
            raise ValueError('Reference dependency version changed: ' + name)
        record = package.read_text('RECORD')
        if not record:
            raise ValueError('Installed dependency RECORD missing')
        dependencies[name] = dict(version=package.version, license_metadata=package.metadata.get('License'),
                                  record_sha256=hashlib.sha256(record.encode()).hexdigest())
    namespace = {}
    exec(compile(source.read_text(), '<pinned-text-reference>', 'exec'), namespace)
    tokenizer = namespace['FastTokenizer']()
    rng = random.Random(839)
    fragments = ['Café', 'RUNNING', 'sensor', 'reaction', '12.5', '-x-', '--', '&', ':)', '(:', ':-)',
                 'αβ', '漢字', '🙂', '\u0301', '\t', '\n', ' ', '!!', 'bbaa']
    examples = ['', 'Café running :)', '\t\n\r', 'bbaa', ' A -- B ', 'a' * 1000]
    examples += [''.join(rng.choices(fragments, k=rng.randrange(1, 30))) for _ in range(256)]
    cases = 0
    for text in examples:
        for mode in (False, True):
            expected = namespace['clean_text'](text, tokenizer, hashchars=mode)
            assert clean_source_text(text, hashchars=mode) == expected
            cases += 1
    files = ['sciona/mercari_text_cleaning.py', 'sciona/mercari_text_policy.py',
             'sciona/character_run_tokenizer.py', 'tests/test_mercari_text_cleaning.py',
             'scripts/validate_mercari_text_cleaning.py', 'requirements/mercari-text-reference.txt',
             'docs/licenses/mercari-solution-LICENSE.txt']
    return dict(format='mercari-text-cleaning-reference.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, exact_reference_cases=cases, worker_processes=0,
        source_commit='374301ad1c32cbc93dcc40313d5d7bb9c5503746', source_sha256=SOURCE_SHA,
        runtime_dependencies=dependencies,
        software_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        limits=['Same-runtime source comparison; historical transliteration tables and NLTK defaults are not recovered or proven equivalent.',
                'Dependency license/publication review remains open; installed metadata is recorded without asserting permission for distribution.',
                'Lossy ASCII transliteration and English stemming require an explicit application policy; do not apply automatically to arbitrary languages or symbolic scientific text.',
                'Input is text only; missing-value coercion, trimming and categorical handling belong to the still-unqualified table adapter.',
                'No full preprocessing graph, trained ensemble or original CDG approval is established.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    result = validate(parser.parse_args().source)
    (ROOT/'docs/reviews/competition_mercari_text_cleaning_reference.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, exact_reference_cases=result['exact_reference_cases'], approved=False)))
