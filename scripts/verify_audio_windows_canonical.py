"""Fresh canonical/legacy serving, document contracts and callable execution."""
import importlib
import json
import numpy as np
from scripts.promote_audio_windows_canonical import (
    ROOT, ARTIFACT, VERSION, FQDN, RUNTIME, RUNNER, PORTS,
    connect, check, qualification, sha, require,
)


def selection():
    report = qualification()
    with connect(True) as db:
        evidence = db.execute('SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s', (VERSION,RUNNER)).fetchone()
        require(evidence is not None, 'Canonical approval missing')
        legacy = evidence['details']['legacy_identity_sha256']
        check(db,report,legacy)
        document = db.execute('SELECT get_artifact_document(%s) AS d', (FQDN,)).fetchone()['d']
        require(document['artifact']['artifact_id'] == ARTIFACT and document['artifact']['status'] == 'approved' and document['artifact']['is_publishable'], 'Served document state differs')
        ports = sorted(document['io_specs'],key=lambda p:(p['direction'],p['ordinal']))
        require(len(ports) == 4 and all(p['version_id'] == VERSION for p in ports), 'Served document port versions differ')
        require([{k:p[k] for k in PORTS[0]} for p in ports] == PORTS, 'Served document contract differs')
        selected = db.execute('SELECT a.import_module,a.source_symbol FROM catalog_atoms_served s JOIN atoms a USING(atom_id) WHERE s.atom_id=%s', (ARTIFACT,)).fetchone()
        require(selected['import_module']+'.'+selected['source_symbol'] == RUNTIME, 'Served import identity differs')
    return selected,report,legacy


def verify():
    before = selection()
    selected, numerical, legacy = before
    module = importlib.import_module(selected['import_module'])
    function = getattr(module,selected['source_symbol'])
    require(sha(module.__file__) == numerical['provider_sha256'], 'Selected provider source changed')
    values = np.linspace(-2,3,31,dtype=np.float64)[::-1]
    result = function(values,5,3)
    expected = np.array([[values[i+j] for j in range(5)] for i in range(0,len(values)-4,3)])
    np.testing.assert_array_equal(result,expected)
    require(not result.flags.writeable and np.shares_memory(result,values), 'Selected provider alias contract differs')
    require(selection() == before, 'Serving changed during execution')
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,artifact_id=ARTIFACT,version_id=VERSION,
                canonical_and_legacy_served=True,document_ports_verified=True,selected_callable_executed=True,
                fresh_numerical_cases=numerical['numerical_cases'],fresh_invalid_cases=numerical['invalid_cases'],
                legacy_identity_sha256=legacy,validator_sha256=sha(__file__),
                publisher_sha256=sha(ROOT/'scripts/promote_audio_windows_canonical.py'),
                numerical_validator_sha256=numerical['validator_sha256'],
                limitations=['Provisioned in-process provider selection and execution; HTTP transport and semantic ranking are not covered.',
                             'Read-only output aliases input storage; persistence or transport need not preserve that alias.'])


if __name__ == '__main__':
    report = verify()
    (ROOT/'docs/reviews/audio_windows_canonical_served.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0)))
