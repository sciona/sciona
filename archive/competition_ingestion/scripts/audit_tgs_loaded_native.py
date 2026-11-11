"""Observe macOS native image origins after synthetic TGS operation warmup."""
import ctypes
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import zipfile


def main():
    import cv2
    import h5py
    import numpy as np
    from scipy.spatial import cKDTree
    import torch
    from sciona.tgs_resnext import TGSResNeXt50
    from sciona.tgs_torch_models import TGSResNet34
    torch.set_num_threads(1)
    for model, size in [(TGSResNeXt50(), 224), (TGSResNet34(5), 128)]:
        model.eval()
        with torch.no_grad():
            assert torch.isfinite(model(torch.rand(1, 3, size, size))).all()
    cv2.resize(np.zeros((101, 101), dtype=np.float32), (192, 192))
    cKDTree(np.eye(3)).query(np.ones((1, 3)), k=2)
    with h5py.File('synthetic-native-probe', 'w', driver='core', backing_store=False) as handle:
        handle.create_dataset('synthetic', data=np.ones(2))
    audit = json.loads(Path('docs/reviews/competition_tgs_publisher_payload_validation.json').read_text())
    assert audit['all_compared_payloads_match'] and audit['all_install_scheme_files_compared']
    ownership = {}
    for name in audit['packages']:
        dist = metadata.distribution(name)
        for entry in dist.files or []:
            path = Path(dist.locate_file(entry)).resolve()
            ownership[path] = (name, str(entry))
    process = ctypes.CDLL(None)
    count = process._dyld_image_count
    count.restype = ctypes.c_uint32
    image_name = process._dyld_get_image_name
    image_name.argtypes = [ctypes.c_uint32]
    image_name.restype = ctypes.c_char_p
    records, system_count, unknown = [], 0, []
    base = Path(sys.base_prefix).resolve()
    artifacts = json.loads(Path('docs/reviews/competition_tgs_published_wheel_audit.json').read_text())['packages']
    authenticated = set()
    for index in range(count()):
        raw = image_name(index).decode()
        if raw.startswith(('/System/', '/usr/lib/')):
            system_count += 1
            continue
        path = Path(raw).resolve()
        if not path.is_file():
            unknown.append(dict(library=path.name, reason='non-system image absent from filesystem'))
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path in ownership:
            package, relative = ownership[path]
            artifact = artifacts[package]['selected_wheel']
            wheel = Path('/private/tmp/sciona_flavours_publisher_wheels') / artifact['filename']
            if package not in authenticated:
                with wheel.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != artifact['sha256']:
                        raise ValueError('publisher wheel drift')
                authenticated.add(package)
            with zipfile.ZipFile(wheel) as archive:
                if relative not in archive.namelist() or hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                    unknown.append(dict(library=path.name, reason='loaded native file differs from publisher wheel'))
                    continue
            records.append(dict(package=package, distribution_file=relative, sha256=digest,
                                publisher_wheel_sha256=audit['packages'][package]['publisher_wheel_sha256'],
                                loaded_file_matches_publisher_payload=True))
        elif path == Path(sys.executable).resolve() or path == base or base in path.parents:
            records.append(dict(platform_interpreter_file=str(path.relative_to(base)) if base in path.parents else path.name,
                                sha256=digest))
        else:
            unknown.append(dict(library=path.name, sha256=digest, reason='outside qualified wheels and declared interpreter/system boundary'))
    report = dict(approved=False, catalog_mutations=0, all_loaded_origins_classified=not unknown,
                  system_images=system_count, native_images=records, unresolved=unknown,
                  python_version=sys.version.split()[0],
                  limits=['Synthetic model, OpenCV, cKDTree and HDF5 warmup only; optional lazy/native branches may load more images.',
                          'Platform system/interpreter binaries are explicitly separate from publisher-wheel authentication.'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path('docs/reviews/competition_tgs_loaded_native.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(native_images=len(records), system_images=system_count, unresolved=unknown)))


if __name__ == '__main__':
    main()
