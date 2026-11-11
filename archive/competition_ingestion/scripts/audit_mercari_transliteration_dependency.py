"""Verify the installed transliteration dependency against its published wheel.

This records software provenance and policy behavior, not distribution permission.
It neither changes catalog license status nor substitutes a different transliterator.
"""
import base64
import csv
import hashlib
import io
import json
from importlib.metadata import distribution
from pathlib import Path
import urllib.request
import zipfile

from sciona.license_policy import LicensePolicy

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit():
    package = distribution('Unidecode')
    if package.version != '1.4.0':
        raise ValueError('Qualified transliteration version required')
    release_url = 'https://pypi.org/pypi/Unidecode/1.4.0/json'
    with urllib.request.urlopen(release_url, timeout=30) as response:
        release = json.load(response)
    wheels = [row for row in release['urls'] if row['filename'] == 'Unidecode-1.4.0-py3-none-any.whl']
    if len(wheels) != 1:
        raise ValueError('Unique published wheel required')
    wheel = wheels[0]
    with urllib.request.urlopen(wheel['url'], timeout=30) as response:
        raw = response.read()
    if sha(raw) != wheel['digests']['sha256']:
        raise ValueError('Published wheel digest mismatch')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    record = archive.read('Unidecode-1.4.0.dist-info/RECORD').decode()
    checked = 0
    for name, digest, size in csv.reader(io.StringIO(record)):
        if not digest:
            if name != 'Unidecode-1.4.0.dist-info/RECORD':
                raise ValueError('Unexpected unhashed wheel member')
            continue
        algorithm, encoded = digest.split('=', 1)
        if algorithm != 'sha256':
            raise ValueError('SHA256 wheel record required')
        content = archive.read(name)
        if (base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=') != encoded
                or len(content) != int(size)):
            raise ValueError('Wheel member integrity mismatch')
        if package.locate_file(name).read_bytes() != content:
            raise ValueError('Installed package differs from published wheel')
        checked += 1
    metadata = archive.read('Unidecode-1.4.0.dist-info/METADATA')
    license_text = archive.read('Unidecode-1.4.0.dist-info/LICENSE')
    classifier = 'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)'
    if classifier not in package.metadata.get_all('Classifier', []):
        raise ValueError('Published GPL-2.0-or-later declaration required')
    if b'GNU GENERAL PUBLIC LICENSE' not in license_text or b'Version 2, June 1991' not in license_text:
        raise ValueError('Expected license text missing')
    row = dict(license_expression='GPL-2.0-or-later', license_family='strong_copyleft')
    behavior = dict(
        unknown_status_default_permitted=LicensePolicy().permits(**row, license_status='unknown'),
        hypothetical_approved_default_permitted=LicensePolicy().permits(**row, license_status='approved'),
        hypothetical_approved_permissive_only_permitted=LicensePolicy(allowed_families=('permissive',)).permits(
            **row, license_status='approved'))
    if behavior != dict(unknown_status_default_permitted=False,
                        hypothetical_approved_default_permitted=True,
                        hypothetical_approved_permissive_only_permitted=False):
        raise ValueError('License policy behavior changed')
    paths = ['scripts/audit_mercari_transliteration_dependency.py', 'sciona/license_policy.py',
             'requirements/mercari-text-reference.txt', 'sciona/mercari_text_cleaning.py']
    return dict(format='mercari-transliteration-dependency-audit.v1', passed=True,
        approved=False, catalog_mutations=0, package='Unidecode', version=package.version,
        source_url='https://pypi.org/project/Unidecode/1.4.0/', release_api_url=release_url,
        published_wheel_sha256=sha(raw), published_wheel_record_sha256=sha(record.encode()),
        installed_members_equal_to_published_wheel=checked,
        metadata_sha256=sha(metadata), license_text_sha256=sha(license_text),
        declared_license='GPL-2.0-or-later', license_family='strong_copyleft',
        declaration_basis='Published package classifier and copyright grant; bundled GPL version 2 text.',
        policy_behavior=behavior,
        software_sha256={p: sha((ROOT/p).read_bytes()) for p in paths},
        findings=['The dependency license declaration is now verified beyond the generic installed GPL label.',
            'The default matcher policy does not categorically prohibit GPL; it requires an approved version status.',
            'A permissive-only principal still rejects this dependency family even if status is approved.',
            'No evidence here establishes distribution compatibility for a combined deliverable or authorizes changing license status.',
            'Retain the current implementation for parity. A different transliterator requires new semantic and full-graph qualification.'],
        remaining_gate='Choose and qualify the concrete distribution/licensing arrangement before approving the application text adapter.',
        reusable_scope='Generic tokenizer, dictionary/text vectorizer state, table composition and neural primitives do not require this application transliteration policy.')


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/mercari_transliteration_dependency.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('passed', 'approved', 'declared_license',
                                                 'installed_members_equal_to_published_wheel')}))
