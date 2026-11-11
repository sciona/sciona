"""Verify the provisioned OpenVaccine worker's declared dependency closure."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]


def dependency_closure():
    pending=['tensorflow','tf-keras','numpy','pandas'];seen={}
    while pending:
        name=canonicalize_name(pending.pop())
        if name in seen:continue
        distribution=metadata.distribution(name);seen[name]=distribution.version
        for raw in distribution.requires or []:
            requirement=Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
            version=metadata.version(requirement.name)
            if not requirement.specifier.contains(version,prereleases=True):
                raise ValueError('Incompatible OpenVaccine dependency: '+requirement.name)
            pending.append(requirement.name)
    return seen


def validate():
    if os.environ.get('TF_USE_LEGACY_KERAS')!='1':raise ValueError('Legacy Keras must be selected before process startup')
    if platform.system()!='Darwin' or platform.machine()!='arm64' or sys.version_info[:2]!=(3,13):
        raise ValueError('This evidence supports only the reviewed macOS arm64 Python 3.13 worker')
    expected=dict(line.split('==') for line in (ROOT/'requirements/openvaccine-execution.txt').read_text().splitlines())
    actual=dependency_closure()
    if expected!=actual:raise ValueError('Worker dependency closure differs from reviewed versions')
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tf_keras
    if (np.__version__,pd.__version__,tf.__version__,tf_keras.__version__)!=tuple(expected[n] for n in ('numpy','pandas','tensorflow','tf-keras')):
        raise ValueError('Imported modules differ from dependency metadata')
    if tf.keras.layers.Layer.__module__.split('.')[0]!='tf_keras':raise ValueError('Legacy Keras selection inactive')
    from sciona.openvaccine_models import SOURCE_SHA256,LICENSE_SHA256
    from sciona.openvaccine_folding import BINARY_SHA256,PARAMETERS_SHA256
    source=Path(os.environ['SCIONA_OPENVACCINE_SOURCE_DIR'])
    dependencies=[(source/'scripts/nullrecurrent_inference.py',SOURCE_SHA256),(source/'LICENSE',LICENSE_SHA256),
        (Path(os.environ['SCIONA_OPENVACCINE_FOLD_BINARY']),BINARY_SHA256),
        (Path(os.environ['SCIONA_OPENVACCINE_FOLD_PARAMETERS']),PARAMETERS_SHA256)]
    for path,digest in dependencies:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Provisioned source or folding dependency changed')
    return actual


def main():
    versions=validate()
    paths=['requirements/openvaccine-execution.txt','scripts/validate_openvaccine_environment.py','docs/reviews/competition_openvaccine_graph_execution.json']
    report=dict(status='passed',dependency_versions=versions,declared_dependency_closure_compatible=True,
        actual_import_versions_match=True,legacy_keras=True,software_dependencies_verified=4,
        platform='macOS arm64',python='3.13',
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits=['Provisioned process environment qualified; not a clean-install, GPU or cross-platform qualification.',
                'Shared runner dependencies are covered by the full graph execution; unrelated installed package constraints are not certified.',
                'Use a dedicated process for this dependency overlay and global initialization seed. No concurrent RNG isolation claim.'])
    (ROOT/'docs/reviews/competition_openvaccine_environment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',dependency_packages=len(versions),software_dependencies_verified=4)))


if __name__=='__main__':main()
