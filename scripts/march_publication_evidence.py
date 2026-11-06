"""Validate immutable March qualification without depending on draft status."""
import importlib
import inspect
import json
import subprocess
from pathlib import Path

from scripts.validate_decomposed_march_baseline import ROOT, sha, runtime

EVIDENCE_COMMIT = 'd97e2cfdc8c3511e145fd4ab3a861eb48ef09398'
EXECUTION = 'docs/reviews/march_corrected_intake_execution.json'
SOURCE = 'docs/reviews/march_corrected_source_review.json'


def strict_equal(actual, expected):
    """JSON comparison that rejects bool/int and int/float substitution."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            strict_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def committed_report(path):
    raw = subprocess.check_output(
        ['git', 'show', EVIDENCE_COMMIT + ':' + path], cwd=ROOT,
        stderr=subprocess.PIPE)
    if raw != (ROOT / path).read_bytes():
        raise ValueError('Immutable qualification changed: ' + path)
    return json.loads(raw)


def validate_reports(execution, source, expected_execution, expected_source):
    # Entire reports are fixed, including coverage, types, limits and hash keys.
    if not strict_equal(execution, expected_execution):
        raise ValueError('Stored execution qualification differs')
    if not strict_equal(source, expected_source):
        raise ValueError('Source qualification differs')
    if execution['passed'] is not True or source['passed'] is not True:
        raise ValueError('Passing qualification required')
    if execution['version_id'] != source['candidate_version_id'] or (
        execution['serialized_graph_sha256'] != source['graph_sha256']):
        raise ValueError('Source and execution identities differ')


def review():
    execution = committed_report(EXECUTION)
    source = committed_report(SOURCE)
    validate_reports(execution, source, execution, source)
    for report in (execution, source):
        for section in ('code_sha256', 'evidence_sha256'):
            for name, digest in report.get(section, {}).items():
                if sha(ROOT / name) != digest:
                    raise ValueError('Qualified file changed: ' + name)
    if sha(ROOT / 'scripts/review_march_corrected_sources.py') != source['reviewer_sha256']:
        raise ValueError('Source reviewer changed')
    for name, digest in execution['provider_sha256'].items():
        module, attribute = name.rsplit('.', 1)
        function = inspect.unwrap(getattr(importlib.import_module(module), attribute))
        if sha(Path(inspect.getfile(function))) != digest:
            raise ValueError('Qualified provider changed: ' + name)
    if not strict_equal(runtime(), execution['runtime']):
        raise ValueError('Qualified runtime changed')
    return dict(review_source='automated', proposed_tier=3,
                verdict='acceptable_with_limits', evidence_commit=EVIDENCE_COMMIT,
                execution_sha256=sha(ROOT / EXECUTION),
                source_review_sha256=sha(ROOT / SOURCE),
                limitations=source['limitations'])


if __name__ == '__main__':
    result = review()
    print(json.dumps(dict(passed=True, proposed_tier=result['proposed_tier'],
                          catalog_mutations=0)))
