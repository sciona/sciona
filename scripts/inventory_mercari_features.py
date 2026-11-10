"""Inventory both pinned feature graphs without loading data or legacy code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'mercari/datasets_tf.py': '2023f6fe3b6a7f21b8372cf077f4805a29356a1c0f190a0caa101b7754e20e8e',
    'mercari/datasets_mx.py': '9befc4fe71e05876170a0df675b234211402bfb9660f47130759bf36a23d52bb',
    'mercari/transformers.py': '3dcc0f896e965bf0acece4cbed61fc58b7d82d10be3bb8a71eb768f747466c44',
    'mercari/text_utils.py': '076683d8877644cf7b8f13b125e57ed596bd2a5455f531bb8f162b4df285c9bc',
}


def operation(node):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError('Expected explicit feature constructor')
    name = node.func.id
    result = dict(operation=name)
    if name in ('make_pipeline', 'make_union_mp'):
        result['children'] = [operation(arg) for arg in node.args]
    elif name == 'FeatureUnionMP':
        if len(node.args) != 1 or not isinstance(node.args[0], ast.List):
            raise ValueError('Expected explicit union branches')
        result['children'] = [operation(pair.elts[1]) for pair in node.args[0].elts]
    # Only algorithm controls are exported; application field bindings remain
    # in the public source and must be implemented as a separate adapter.
    controls = {}
    for keyword in node.keywords:
        if keyword.arg in ('ngram_range', 'binary', 'min_df', 'max_features', 'analyzer', 'dtype', 'n_jobs'):
            try:
                controls[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                controls[keyword.arg] = dict(source_expression=ast.unparse(keyword.value))
    if name in ('SparsityFilter', 'SGDFeatureSelectionV2'):
        controls['threshold'] = ast.literal_eval(node.args[0])
    if controls:
        result['explicit_controls'] = controls
    return result


def inventory(directory):
    for name, digest in PINS.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned feature source changed: ' + name)
    graphs = {}
    for backend, filename in [('tensorflow', 'datasets_tf.py'), ('mxnet', 'datasets_mx.py')]:
        tree = ast.parse((directory/'mercari'/filename).read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('prepare_vectorizer_')]
        if len(functions) != 3:
            raise ValueError('Expected three feature families per backend')
        graphs[backend] = []
        for group, function in enumerate(functions, 1):
            assignments = [n for n in function.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'vectorizer' for t in n.targets)]
            if len(assignments) != 1:
                raise ValueError('Expected one explicit pipeline assignment')
            graph = operation(assignments[0].value)
            union = [n for n in graph['children'] if n['operation'] in ('FeatureUnionMP', 'make_union_mp')]
            if len(union) != 1:
                raise ValueError('Expected one outer feature union')
            graphs[backend].append(dict(group=group, branch_count=len(union[0]['children']), graph=graph))
    assert [g['branch_count'] for g in graphs['tensorflow']] == [6, 4, 4]
    assert [g['branch_count'] for g in graphs['mxnet']] == [12, 5, 5]
    return dict(format='mercari-feature-topology.v1', approved=False, catalog_mutations=0,
        source_commit='374301ad1c32cbc93dcc40313d5d7bb9c5503746', source_sha256=PINS,
        source_repository='https://github.com/pjankiewicz/mercari-solution', variants=graphs,
        reuse_boundaries=[
            'Generic sparse count/TF-IDF encoding, feature concatenation, support filtering and supervised selection can have domain-independent contracts.',
            'Text cleaning, fixed field bindings, categorical conventions, numeric-unit extraction and brand heuristics require explicit application adapters.',
            'Learn vocabularies, scaling, support masks, supervised feature masks and clipping bounds on training inputs; reuse fitted state at inference.',
            'The MXNet feature graph cannot be replaced by the smaller TensorFlow graph while claiming reproduction of the MXNet pipeline.'],
        qualification_gaps=[
            'Topology only: tokenizer expressions, constructor defaults, categorical coercions and feature values are not runtime-qualified.',
            'MXNet group three applies supervised SGD selection at percentile 70 after its bigram encoder; TensorFlow does not.',
            'Source sanitizer fits nanmax over stored sparse values, replaces NaNs with zero, then clips to the fitted maximum; empty/nonfinite bounds require explicit qualification.',
            'Source pattern transformer suppresses digit-containing strings before numeric extraction, contains adjacent concatenated regex literals, and uses removed pandas ix; do not silently repair and claim historical parity.',
            'Legacy sklearn private imports and historical SGD defaults require compatibility qualification.',
            'This inventory does not approve the original CDG or establish complete ensemble execution.'],
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    result = inventory(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_mercari_feature_topology.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: [g['branch_count'] for g in v] for k, v in result['variants'].items()}))
