"""Run frozen synthetic combined-graph checks against relocated ML runtimes."""
import argparse
import ast
import asyncio
import base64
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from scripts.audit_osic_training_finalization import ROOT, REVISION
from scripts.audit_osic_opencv_isolation import audit_distribution, digest

GRAPHS = ('osic_image_lifecycle_graphs', 'osic_image_slice_graph',
          'osic_image_population_graph', 'osic_quantile_replay_graph',
          'osic_keyed_quantile_graph', 'osic_combined_inference_graph')


def frozen(name):
    return subprocess.check_output(['git', 'show', REVISION+':'+name], cwd=ROOT)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_graphs(directory):
    package = directory / 'frozen_osic_graphs'
    package.mkdir()
    (package / '__init__.py').write_text('')
    identities = {}
    for name in GRAPHS:
        raw = frozen('sciona/'+name+'.py')
        identities[name] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {'sciona.'+g for g in GRAPHS}:
                node.module = 'frozen_osic_graphs.'+node.module.split('.')[-1]
        (package / (name+'.py')).write_text(ast.unparse(tree)+'\n')
    return identities


def software():
    # Snapshot source before model work, including transitive relocated helpers.
    paths = list((ROOT.parent/'sciona-atoms-ml/src/sciona/_runtime/ml').glob('*.py'))
    paths += list((ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml').rglob('*.py'))
    paths += list((ROOT.parent/'sciona-atoms/src/sciona/atoms/numerical').rglob('*.py'))
    paths += [ROOT/'sciona/visualizer/runner.py', ROOT/'sciona/services/execution_graph_codec.py',
              ROOT/'sciona/architect/models.py', ROOT/'sciona/architect/handoff.py', Path(__file__)]
    return {str(p.relative_to(ROOT.parent)):sha(p) for p in sorted(paths)}


def worker(state, overlay_path):
    from frozen_osic_graphs.osic_combined_inference_graph import build_combined_inference_graph
    from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
    before = software()
    tree = ast.parse(frozen('scripts/validate_osic_combined_inference_graph.py'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker']
    if len(functions) != 1:
        raise ValueError('Expected one frozen worker')
    edits = []
    for node in ast.walk(functions[0]):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == 'sciona.osic_image_population':
            edits.append(node.module)
            node.module = 'sciona._runtime.ml.osic_image_population'
        elif node.module == 'sciona' and [n.name for n in node.names] == [
                'osic_feature_state','osic_quantile_features','osic_quantile_training']:
            edits.append('sciona quantile runtime imports')
            node.module = 'sciona._runtime.ml'
        elif node.module == 'scripts.validate_osic_image_state':
            edits.append(node.module)
            node.module = 'frozen_osic_graphs.coordinates'
    if len(edits) != 3:
        raise ValueError('Unexpected frozen worker import contract')
    namespace = dict(globals(), overlay=SimpleNamespace(sha=sha),
                     build_combined_inference_graph=build_combined_inference_graph,
                     encode_execution_graph=encode_execution_graph, decode_execution_graph=decode_execution_graph)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen-combined-worker>', 'exec'), namespace)
    execution = namespace['worker'](state, overlay_path)
    graph_hash, nodes, edges = encode_execution_graph(build_combined_inference_graph())
    if graph_hash != '586bf24fb2947aad1a1c3e3b20a5348679062922ad247c7c908135df22eede4d':
        raise ValueError('Frozen graph identity changed')
    for index in range(len(edges)):
        try:
            decode_execution_graph(nodes, edges[:index]+edges[index+1:], graph_hash)
        except ValueError:
            pass
        else:
            raise ValueError('Missing edge accepted')
    if software() != before:
        raise ValueError('Current software changed during qualification')
    loaded = {str(Path(module.__file__).resolve()) for module in tuple(sys.modules.values())
              if getattr(module, '__file__', None)}
    used = {name:value for name,value in before.items()
            if str((ROOT.parent/name).resolve()) in loaded}
    return dict(execution=execution, current_software_sha256=used,
                import_relocations=edits, missing_edges_rejected=len(edges))


def validate(dependencies, state):
    tree = ast.parse(frozen('scripts/validate_osic_image_model.py'))
    pins = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PINS' for t in n.targets))
    pins['pydicom-3.0.1-py3-none-any.whl'] = 'db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41'
    if any(sha(dependencies/name) != expected for name, expected in pins.items()):
        raise ValueError('Pinned dependencies differ')
    cv_before = audit_distribution('opencv-python-headless')
    if cv_before['missing'] or cv_before['mismatched']:
        raise ValueError('OpenCV distribution cannot be isolated')
    with tempfile.TemporaryDirectory(prefix='sciona-osic-relocated-') as temporary:
        overlay_path = Path(temporary)
        graphs = stage_graphs(overlay_path)
        tree = ast.parse(frozen('scripts/validate_osic_image_state.py'))
        coordinates = [n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'COORDINATES' for t in n.targets)]
        if len(coordinates) != 1:
            raise ValueError('Expected one frozen coordinate fixture')
        (overlay_path/'frozen_osic_graphs/coordinates.py').write_text(ast.unparse(coordinates[0])+'\n')
        for name in pins:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies/name) as archive:
                    archive.extractall(overlay_path)
        distribution = metadata.distribution('opencv-python-headless')
        copied = {}
        for entry in distribution.files or []:
            relative = Path(str(entry))
            if not entry.hash or relative.parts[0] != 'cv2':
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe package path')
            target = overlay_path/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(distribution.locate_file(entry), target)
            if base64.urlsafe_b64encode(digest(target, entry.hash.mode)).rstrip(b'=').decode() != entry.hash.value:
                raise ValueError('Copied OpenCV differs from RECORD')
            copied[str(relative)] = sha(target)
        env = dict(os.environ, PYTHONPATH=str(overlay_path)+os.pathsep+str(ROOT),
                   TF_USE_LEGACY_KERAS='1', TF_CPP_MIN_LOG_LEVEL='3', PYTHONDONTWRITEBYTECODE='1',
                   MPLCONFIGDIR='/private/tmp/sciona-mpl-cache')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
            env[key] = '1'
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
            '--dependencies', str(overlay_path), '--state-directory', str(state)], env=env,
            capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError('Relocated qualification failed:\n'+process.stderr)
        result = json.loads(process.stdout)
        if copied != {name:sha(overlay_path/name) for name in copied}:
            raise ValueError('OpenCV overlay changed')
    if audit_distribution('opencv-python-headless') != cv_before:
        raise ValueError('Shared OpenCV distribution changed')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
                frozen_revision=REVISION, frozen_graph_sha256=graphs,
                dependency_archive_sha256=pins, **result,
                limitations=[
                    'Frozen graph builders in a temporary namespace; this does not install graph builders into the migrated runtime.',
                    'Numerical checks use current relocated providers, an intermediate image checkpoint, and two fresh synthetic quantile fits.',
                    'No full image-training, original-CDG approval, empirical effectiveness or independent source-notebook parity claim.',
                    'Explicit registry discovery and intercepted persistence; default discovery, HTTP transport and caching remain unqualified.',
                    'No catalog approvals are renewed by this check.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--state-directory', type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.state_directory, args.dependencies)))
    else:
        result = validate(args.dependencies, args.state_directory)
        (ROOT/'docs/reviews/osic_relocated_combined_graph.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['execution'], indent=2))
