"""Pin transitive local provider source and retain installed numerical notices."""
import ast
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record():
    review_path = ROOT/'docs/reviews/residual_classifier_binding_review.json'
    review = json.loads(review_path.read_text())
    if not review['passed']:
        raise ValueError('Passing binding preflight required')
    pending = [b['runtime'].rsplit('.', 1)[0] for b in review['bindings']]
    sources = {}
    provider_root = (ROOT.parent/'sciona-atoms-ml/src').resolve()
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if provider_root not in path.parents:
            raise ValueError('Provider source outside expected checkout')
        sources[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('sciona.atoms.ml.'):
                pending.append(node.module)
            elif isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names if alias.name.startswith('sciona.atoms.ml.'))
    for binding in review['bindings']:
        if sources[binding['runtime'].rsplit('.', 1)[0]] != binding['source_sha256']:
            raise ValueError('Reviewed provider binding drift')
    destination = ROOT/'docs/licenses/residual-classifier'
    notices = {}
    for name in ['xgboost', 'pandas', 'scikit-learn', 'scipy']:
        distribution = metadata.distribution(name)
        retained = []
        for item in distribution.files or []:
            if not any(part.lower() in {'licenses','license','license.txt','copying','copying.txt','notice','notice.txt'} for part in Path(item).parts):
                continue
            relative = Path(item)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unexpected notice location')
            source = Path(distribution.locate_file(item))
            if not source.is_file():
                continue
            target = destination/name/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError('Existing retained notice differs')
            target.write_bytes(source.read_bytes())
            retained.append(dict(notice=str(target.relative_to(ROOT)), sha256=sha(target)))
        if not retained:
            raise ValueError('No installed notices found: '+name)
        notices[name] = dict(version=distribution.version, retained=retained)
    return dict(passed=True, approved=False, catalog_mutations=0,
        binding_review_sha256=sha(review_path), provider_source_sha256=dict(sorted(sources.items())),
        retained_notices=notices, recorder_sha256=sha(Path(__file__)),
        limitations=['Local provider imports recursively pinned; third-party software versions covered by the separate dependency review.',
            'Installed numerical-package notices retained; this is not a legal determination or a complete dependency license audit.',
            'Shared source and NumPy notices are retained in docs/licenses/conditional-correction.',
            'Framework source, runtime evidence binding, remaining dependency notices and publication transactions still require review.'])


if __name__ == '__main__':
    result = record()
    (ROOT/'docs/reviews/residual_classifier_provider_dependencies.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, provider_modules=len(result['provider_source_sha256']),
        numerical_packages=len(result['retained_notices']), retained_notice_files=sum(len(v['retained']) for v in result['retained_notices'].values()), catalog_mutations=0)))
