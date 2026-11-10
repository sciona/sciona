"""Recheck full Cassava prerequisites without launching training or changing pins."""
import hashlib
import json
from pathlib import Path
import requests
from scripts.import_flavours_provider_drafts import ROOT
from scripts.audit_decomposed_cassava_readiness import audit


def check():
    current=audit()
    pin_path=ROOT/'docs/reviews/competition_cassava_cropnet_artifact.json'
    pin=json.loads(pin_path.read_text())
    directory=Path('/private/tmp/sciona_cassava_cropnet_reference/model')
    matching=0
    for name,digest in pin['files'].items():
        path=directory/name
        if path.is_file():
            with path.open('rb') as stream:
                matching+=hashlib.file_digest(stream,'sha256').hexdigest()==digest
    urls=[pin['reference'],'https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2?tf-hub-format=compressed']
    endpoints=[]
    for url in urls:
        with requests.get(url,stream=True,timeout=30) as response:
            endpoints.append(dict(url=url,http_status=response.status_code,content_type=response.headers.get('Content-Type'),
                                  artifact_downloaded=False))
    return dict(format='cassava-artifact-prerequisites.v1',approved=False,catalog_mutations=0,training_launched=False,
        cropnet_local_files=dict(expected=len(pin['files']),matching=matching),endpoints=endpoints,
        cropnet_pin_sha256=hashlib.sha256(pin_path.read_bytes()).hexdigest(),
        current_runtime_readiness=current,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remaining=['Restore all exact reviewed CropNet files before the complete four-family execution.',
                   'Requalify current runtime and execute all sixteen trainable fits and frozen CropNet inference.',
                   'Provider contracts, original staging, publication and served selection remain required.'],
        limits=['HTTP status is an availability observation, not proof that a different endpoint or local cache cannot exist.',
                'No alternative model, omitted branch, reduced fit budget or changed artifact pin is accepted by this check.',
                'No private runtime images or task records are accessed.'])


if __name__=='__main__':
    report=check()
    (ROOT/'docs/reviews/cassava_artifact_prerequisites_current.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cropnet_local_files=report['cropnet_local_files'],
                         endpoint_statuses=[r['http_status'] for r in report['endpoints']],
                         package_mismatches=len(report['current_runtime_readiness']['package_mismatches']),training_launched=False)))
