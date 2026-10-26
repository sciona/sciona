"""Retain distributed notices for the native software found by link traversal.

No binaries or runtime inputs are copied. This records notice provenance;
it is not a legal opinion or a clean-install qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
GLIB_SHA='93c941aa17d5eb1d53fe838365f29a8b4e539c222a256d974ec8f30fc413e396'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--glib-source',required=True,type=Path)
    cli=parser.parse_args();links_path=ROOT/'docs/reviews/competition_otto_candidate_native_links.json'
    links=json.loads(links_path.read_text());licenses={};origins={};packages={}
    def retain(origin,data):
        digest=sha(data);name='Otto-native-notice-'+digest[:16]+'.txt'
        target=ROOT/'docs/licenses'/name
        if target.exists() and target.read_bytes()!=data:raise ValueError('Retained notice collision')
        if not target.exists():target.write_bytes(data)
        licenses[name]=digest;origins[origin]=name
    software_roots=set()
    for file in links['native_files']:
        parts=Path(file).parts
        if 'Cellar' in parts:
            index=parts.index('Cellar');software_roots.add(Path(*parts[:index+3]))
    for root in sorted(software_roots):
        label=root.parent.name+'/'+root.name;notices=[]
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.name.upper().startswith(('LICENSE','COPYING','NOTICE','COPYRIGHT')) and path.suffix not in ('.class','.py','.pyc','.h','.db'):
                origin=label+'/'+str(path.relative_to(root));retain(origin,path.read_bytes());notices.append(origin)
        formula=root/'.brew'/(root.parent.name+'.rb')
        metadata=dict(notice_origins=notices,formula_sha256=sha(formula.read_bytes()))
        text=formula.read_text()
        match=re.search(r'^  license (.+)$',text,re.M)
        metadata['formula_license_declaration']=match.group(1) if match else 'See hash-bound formula'
        if root.parent.name=='glib':
            if sha(cli.glib_source.read_bytes())!=GLIB_SHA or GLIB_SHA not in text:raise ValueError('GLib source identity differs')
            with tarfile.open(cli.glib_source) as archive:
                for member in archive.getmembers():
                    parts=Path(member.name).parts
                    if member.isfile() and (('LICENSES' in parts) or Path(member.name).name.upper().startswith(('COPYING','NOTICE','COPYRIGHT'))):
                        origin=label+'/source/'+str(Path(*parts[1:]));retain(origin,archive.extractfile(member).read());notices.append(origin)
            metadata.update(source_sha256=GLIB_SHA,source_url='https://download.gnome.org/sources/glib/2.90/glib-2.90.1.tar.xz')
        if not notices:raise ValueError('Missing distributed notices: '+label)
        packages[label]=metadata
    python_license=Path(sys.base_prefix)/'lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'LICENSE.txt'
    retain('Python/'+sys.version.split()[0]+'/LICENSE.txt',python_license.read_bytes())
    report=dict(status='passed',qualification='candidate_notices_only',approved=False,catalog_mutations=0,homebrew_packages=packages,
        license_sha256=licenses,license_origins=origins,
        native_link_evidence_sha256=sha(links_path.read_bytes()),verifier_sha256=sha(Path(__file__).read_bytes()),
        scope='Distributed notices for all Homebrew roots in the verified native link graph and the base Python interpreter. Algorithm package, R extension and source-checkout notices are retained in the separate environment inventory.',
        limits=['No software binaries or source archives are redistributed by this report.',
                'macOS-provided shared-cache libraries remain operating-system prerequisites.',
                'Notice retention does not establish clean installation, historical reproduction or competitive accuracy.'])
    (ROOT/'docs/reviews/competition_otto_candidate_native_notices.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',homebrew_packages=len(packages),distinct_notices=len(licenses),notice_origins=len(origins))))


if __name__=='__main__':main()
