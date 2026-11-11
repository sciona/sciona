"""Replay the immutable numerical qualification with the reviewed graph schema."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def qualified_program():
    review_path = ROOT / 'docs/reviews/osic_graph_schema_transition.json'
    raw_review = subprocess.check_output([
        'git', 'show', 'f14a4355:docs/reviews/osic_graph_schema_transition.json'], cwd=ROOT)
    if raw_review != review_path.read_bytes():
        raise ValueError('Immutable schema review changed')
    review = json.loads(raw_review)
    if not review['passed'] or not review['historical_envelope_accepted']:
        raise ValueError('Successful schema compatibility review required')
    for name, digest in review['current_source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Reviewed graph software changed')
    original = subprocess.check_output([
        'git', 'show', 'f28f2dde:scripts/qualify_osic_relocated_combined_graph.py'], cwd=ROOT)
    tree = ast.parse(original)
    replacements = {
        review['historical_graph_sha256']: review['current_graph_sha256'],
        'docs/reviews/osic_relocated_combined_graph.json':
            'docs/reviews/osic_current_schema_combined_graph.json',
    }
    counts = dict.fromkeys(replacements, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replacements:
            old = node.value
            node.value = replacements[old]
            counts[old] += 1
    if set(counts.values()) != {1}:
        raise ValueError('Unexpected immutable qualifier structure')
    # No worker assertion or numerical operation is altered: only the reviewed
    # schema identity and the distinct output report destination change.
    return compile(tree, '<schema-qualified-numerical-review>', 'exec')


if __name__ == '__main__':
    exec(qualified_program(), dict(__name__='__main__', __file__=__file__))
