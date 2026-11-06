"""Verify installed active numerical dependency closure against RECORD hashes."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
ROOT_PACKAGES=['numpy','torch','scipy','scikit-learn','pandas','sympy','icontract','pydantic','threadpoolctl']


def review():
    pending=list(ROOT_PACKAGES);packages={}
    while pending:
        name=canonicalize_name(pending.pop())
        if name in packages:continue
        dist=metadata.distribution(name);requirements=[]
        for raw in dist.requires or []:
            requirement=Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
            if requirement.extras or requirement.url:raise ValueError('Unreviewed requirement extras or URL: '+name)
            if not requirement.specifier.contains(metadata.version(requirement.name),prereleases=True):
                raise ValueError('Unsatisfied installed dependency: '+name)
            requirements.append(str(requirement));pending.append(requirement.name)
        verified=0
        for relative in dist.files or []:
            if relative.hash is None:continue
            with dist.locate_file(relative).open('rb') as stream:
                digest=hashlib.file_digest(stream,relative.hash.mode).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b'=').decode()!=relative.hash.value:
                raise ValueError('Installed RECORD mismatch: '+name)
            verified+=1
        if not verified:raise ValueError('No hashed installed files: '+name)
        packages[name]=dict(version=dist.version,hashed_files_verified=verified,
            active_requirements=sorted(requirements))
    return dict(format='video-numeric-runtime.v1',passed=True,approved=False,catalog_mutations=0,
        root_packages=ROOT_PACKAGES,installed_packages=dict(sorted(packages.items())),
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Current-platform active base dependency closure only; optional extras are excluded.',
                'Installed RECORD consistency does not authenticate upstream publishers or establish clean-platform reproducibility.',
                'Runtime integrity evidence alone does not approve providers or qualify full CDG semantics.',
                'Provider and matcher source fingerprints are recorded separately in graph execution evidence.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/video_numeric_runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,packages=len(result['installed_packages']),approved=False)))
