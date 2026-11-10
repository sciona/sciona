"""Validate serial generic tokenization against the source character policy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random

from sciona.mercari_text_policy import source_tokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '076683d8877644cf7b8f13b125e57ed596bd2a5455f531bb8f162b4df285c9bc'


def validate(source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Pinned tokenizer source required')
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'FastTokenizer')
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<pinned-tokenizer-reference>', 'exec'), namespace)
    reference, actual = namespace['FastTokenizer'](), source_tokenizer()
    assert actual._words == reference._default_word_chars_set
    assert actual._spaces == reference._default_white_space_set
    alphabet = reference._default_word_chars + ' \t\n\r!?🙂漢字\u0301\u00a0'
    rng = random.Random(833)
    examples = ['', 'A--B 12&3!!', '\t \n\r', '🙂🙂漢字', 'a\u0301', 'a' * 10000]
    examples += [''.join(rng.choices(alphabet, k=rng.randrange(1, 500))) for _ in range(256)]
    for text in examples:
        assert actual(text) == reference(text)
        assert ''.join(actual(text)) == text
    files = ['sciona/character_run_tokenizer.py', 'sciona/mercari_text_policy.py',
             'tests/test_character_run_tokenizer.py', 'scripts/validate_mercari_tokenizer.py',
             'docs/licenses/mercari-solution-LICENSE.txt']
    return dict(format='mercari-tokenizer-reference.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, exact_reference_cases=len(examples), worker_processes=0,
        source_commit='374301ad1c32cbc93dcc40313d5d7bb9c5503746', source_sha256=SOURCE_SHA,
        software_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        limits=['Token boundaries only; separate source cleaning, transliteration, stemming and table preprocessing remain unqualified.',
                'Generic core requires explicit disjoint word and whitespace alphabets; this adapter preserves the source character policy.',
                'Code-point behavior intentionally preserves punctuation and whitespace; it is not linguistic word segmentation or grapheme-cluster handling.',
                'Source PreprocessDataPJ hard-codes Pool(4) despite its n_jobs argument; do not execute that preprocessor unchanged alongside other compute.',
                'No full feature pipeline, training or original CDG approval is claimed.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    result = validate(parser.parse_args().source)
    (ROOT/'docs/reviews/competition_mercari_tokenizer_reference.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, exact_reference_cases=result['exact_reference_cases'], approved=False)))
