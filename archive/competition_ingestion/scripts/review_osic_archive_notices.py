"""Retain exact notices from pinned OSIC source and dependency archives.

This is an archive identity/notice review, not a complete runtime or publication
approval. It neither installs packages nor reads source notebooks or datasets.
"""
import argparse
from email.parser import BytesParser
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from scripts.validate_osic_image_model import PINS
from scripts.validate_osic_dicom import WHEEL,WHEEL_SHA

ROOT=Path(__file__).resolve().parents[1]
NOTICES={
    'efficientnet-1.1.0-py3-none-any.whl':('efficientnet-1.1.0.dist-info/LICENSE',
        'e486cea90cab95bee73cbe32de74e5e8e84ba4e87228e837be71af9e41a55203',
        'Cassava-qubvel-efficientnet-Apache-2.0.txt','Apache-2.0'),
    'Keras_Applications-1.0.8-py3-none-any.whl':('Keras_Applications-1.0.8.dist-info/LICENSE',
        '82e4461dcead03930115f469fd487dc1e60421119e36b25c5be707f1dba08170',
        'OSIC-Keras-Applications-MIT.txt','MIT'),
    'pydicom-3.0.1-py3-none-any.whl':('pydicom-3.0.1.dist-info/LICENSE',
        '39b3259cb15151df25167119087df1fef33624765e639af42447ba6fe86dada3',
        'OSIC-pydicom-MIT.txt','MIT')}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def retain(raw,name):
    destination=ROOT/'docs/licenses'/name
    if destination.exists() and destination.read_bytes()!=raw:
        raise ValueError('Existing retained notice differs: '+name)
    if not destination.exists():
        destination.write_bytes(raw)
    return 'docs/licenses/'+name


def review(dependencies,source):
    pins=dict(PINS);pins[WHEEL]=WHEEL_SHA
    for name,digest in pins.items():
        if sha((dependencies/name).read_bytes())!=digest:
            raise ValueError('Pinned archive changed: '+name)
    distributions={};licenses={}
    for name,(member,digest,destination,declaration) in NOTICES.items():
        with ZipFile(dependencies/name) as archive:
            raw=archive.read(member)
            if sha(raw)!=digest:
                raise ValueError('Pinned notice changed')
            fields=BytesParser().parsebytes(archive.read(member.rsplit('/',1)[0]+'/METADATA'))
            normalized=canonicalize_name(fields['Name'])
            distributions[normalized]=(fields['Version'],fields.get_all('Requires-Dist',[]))
            licenses[normalized]=dict(archive=name,archive_sha256=pins[name],version=fields['Version'],
                declared_license=declaration,notice_sha256=digest,retained_notice=retain(raw,destination))
            if normalized=='pydicom':
                # The retained file also contains the GDCM-derived dictionary
                # notice; the distribution's MIT label does not replace it.
                licenses[normalized]['additional_notice']='GDCM-derived private dictionary portions: three-clause BSD terms retained in the same file'
    # The separately pinned Keras-Applications source archive must carry the
    # same notice as the wheel whose Python modules were already compared.
    with ZipFile(dependencies/'keras-applications-source.zip') as archive:
        names=[name for name in archive.namelist() if name.endswith('/LICENSE')]
        if len(names)!=1 or sha(archive.read(names[0]))!=NOTICES['Keras_Applications-1.0.8-py3-none-any.whl'][1]:
            raise ValueError('Source/wheel Keras-Applications notices differ')
    requirements=[]
    for distribution,(_,declared) in distributions.items():
        for text in declared:
            requirement=Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):
                continue
            if requirement.url or requirement.extras:
                raise ValueError('Unreviewed direct reference or extra')
            name=canonicalize_name(requirement.name)
            version=distributions[name][0] if name in distributions else metadata.version(name)
            if not requirement.specifier.contains(version,prereleases=True):
                raise ValueError('Unsatisfied active requirement')
            requirements.append(dict(distribution=distribution,requirement=str(requirement),resolved_version=version,
                resolution='pinned_overlay' if name in distributions else 'provisioned_environment'))
    raw=(source/'LICENSE').read_bytes()
    if sha(raw)!='6b0382b16279f26ff69014300541967a356a666eb0b91b422f6862f6b7dad17e':
        raise ValueError('Pinned original source license changed')
    original=dict(declared_license='Unlicense',notice_sha256=sha(raw),
        retained_notice=retain(raw,'OSIC-source-Unlicense.txt'))
    return dict(format='osic-archive-notices.v1',passed=True,approved=False,catalog_mutations=0,
        package_installations=0,source_notice=original,dependency_notices=licenses,
        keras_applications_source_wheel_notice_equal=True,active_direct_requirements=requirements,
        reviewer_sha256=sha(Path(__file__).read_bytes()),
        limitations=['Exact declared source/archive notices only; not an independent legal opinion or complete redistribution approval.',
            'Installed transitive runtime packages, native library notices and dependency provenance require their own review.',
            'No optional extras are enabled by this review; codec availability and actual plugin selection remain separate runtime checks.',
            'The existing efficientnet notice is reused byte-for-byte; its historical filename does not narrow its applicability.',
            'Full original workflow training, provider registration and publication gates remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies',required=True,type=Path)
    parser.add_argument('--source-directory',required=True,type=Path)
    args=parser.parse_args()
    result=review(args.dependencies,args.source_directory)
    (ROOT/'docs/reviews/osic_archive_notices.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,dependency_notices=len(result['dependency_notices']),
        active_requirements=len(result['active_direct_requirements']),approved=False)))
