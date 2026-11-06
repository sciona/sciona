"""Restore hash-pinned software dependencies only, using one compiler worker."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provision():
    reviews = ROOT/'docs/reviews'
    model = json.loads((reviews/'competition_openvaccine_source_pins.json').read_text())
    engine = json.loads((reviews/'competition_openvaccine_folding_engine.json').read_text())
    coefficients = json.loads((reviews/'competition_openvaccine_eternafold_parameters.json').read_text())
    groups = [
        ('DasLab/KaggleOpenVaccine',model['commit'],Path('/private/tmp/sciona_openvaccine_source'),
         [p for p in model['pins'] if p['software_path'] in ('LICENSE','README.md','requirements.txt','scripts/nullrecurrent_inference.py')]),
        ('csfoo/contrafold-se',engine['engine_commit'],Path('/private/tmp/sciona_openvaccine_contrafold'),engine['source_pins']),
        ('WaymentSteeleLab/EternaFold',coefficients['commit'],Path('/private/tmp/sciona_openvaccine_eternafold_parameters'),
         [dict(software_path=coefficients['software_parameter_path'],sha256=coefficients['sha256']),*coefficients['parameter_license_pins']]),
    ]
    counts = []
    for repository,commit,directory,pins in groups:
        for pin in pins:
            relative = Path(pin['software_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid software pin path')
            path = directory/relative
            if path.exists():
                if sha(path) != pin['sha256']:
                    raise ValueError('Existing software differs from reviewed pin')
                continue
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/{repository}/{commit}/{relative.as_posix()}',timeout=30) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != pin['sha256']:
                raise ValueError('Downloaded software differs from reviewed pin')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
        (directory/'manifest.json').write_text(json.dumps(dict(repository=repository,commit=commit,pins=pins),indent=2)+'\n')
        counts.append(dict(repository=repository,commit=commit,verified_software_files=len(pins)))
    directory = groups[1][2]/'src'
    binary = directory/'contrafold'
    compiler = subprocess.check_output(['clang++','--version'],text=True).splitlines()[0]
    if not binary.exists():
        if compiler != engine['compiler']:
            raise ValueError('Reviewed compiler required for byte-identical rebuild')
        subprocess.run(['clang++',*engine['build_flags'],'Contrafold.cpp','FileDescription.cpp',
            'Options.cpp','SStruct.cpp','Utilities.cpp','-o','contrafold'],cwd=directory,check=True)
    if sha(binary) != engine['binary_sha256']:
        raise ValueError('Rebuilt executable differs from reviewed binary')
    # Confirm compilation did not alter any pinned source file.
    for _,_,directory,pins in groups:
        if any(sha(directory/p['software_path']) != p['sha256'] for p in pins):
            raise ValueError('Software changed while provisioning')
    return dict(passed=True,catalog_mutations=0,approved=False,software=counts,
        binary_sha256=sha(binary),reviewed_binary_exact=True,compiler=compiler,compiler_workers=1,
        provisioner_sha256=sha(__file__),
        scope='Pinned model definitions, licenses, folding source and published folding coefficients only. No sequence dataset, competition predictor checkpoint or notebook output is fetched. Existing mismatched software is rejected. No neural execution or clean-install claim.')


if __name__ == '__main__':
    report = provision()
    (ROOT/'docs/reviews/decomposed_openvaccine_software_provision.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
