"""Prepare exact non-serving character-tokenizer provider identities and port contracts."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import NAMESPACE_URL, uuid5

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_COMMIT = 'df784132'
EVIDENCE = 'docs/reviews/registered_tokenizer_feature_integration.json'
MODULE = 'sciona.atoms.ml.character_run_tokenization'
CONSTRAINTS = {
    'text': 'Unicode Python string; empty input allowed. No coercion of missing values, case folding, normalization, transliteration, stemming or deletion. Caller resolves table missing-value policies before this operation.',
    'word_characters': 'Explicit string whose code points form the word-run alphabet. Empty and repeated characters allowed; membership only matters. Must be disjoint from whitespace_characters. Caller preserves the same policy across training and inference and chooses it for the intended domain.',
    'whitespace_characters': 'Explicit string whose code points form the whitespace-run alphabet. Empty and repeated characters allowed. No automatic Unicode whitespace expansion. Must be disjoint from word_characters; characters outside both alphabets become individual tokens.',
}



def plan():
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT)
    if raw != (ROOT/EVIDENCE).read_bytes():
        raise ValueError('Committed connected execution evidence changed')
    evidence = json.loads(raw)
    expected = {(g, role) for g in (1, 2, 3) for role in ('training', 'calibration', 'query')}
    if (evidence.get('passed') is not True or evidence.get('synthetic_only') is not True
            or evidence['exact_tokenizer_reference_cases'] != 262
            or len(evidence['cases']) != 9
            or {(r['group'], r['synthetic_role']) for r in evidence['cases']} != expected
            or not all(r['exact_canonical_csr_equal'] is True for r in evidence['cases'])
            or evidence['tokenizer_calls_by_group'] != {'1': 67840, '3': 33920, '2': 0}
            or evidence['registered_tokenizer_calls'] != 101760):
        raise ValueError('Exact registered tokenizer and connected feature evidence required')
    for relative, digest in evidence['software_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Execution implementation changed: '+relative)
    repo = ROOT.parent/'sciona-atoms-ml'
    for module, digest in evidence['provider_sha256'].items():
        path = repo/'src'/Path(*module.split('.')).with_suffix('.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Connected provider implementation changed: '+module)
    provider_path = repo/'src/sciona/atoms/ml/character_run_tokenization.py'
    provider = dict(revision='79eaaa3', sha256=evidence['provider_sha256'][MODULE],
                    functions=['tokenize_character_runs'])
    committed = subprocess.check_output(['git', 'show', provider['revision']+':src/sciona/atoms/ml/character_run_tokenization.py'], cwd=repo)
    if committed != provider_path.read_bytes() or hashlib.sha256(committed).hexdigest() != provider['sha256']:
        raise ValueError('Immutable provider implementation changed')
    definitions = {n.name: n for n in ast.parse(committed).body if isinstance(n, ast.FunctionDef)}
    specs = {s.source_symbol: s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE}
    if set(specs) != set(provider['functions']):
        raise ValueError('Exact provider inventory required')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        owners = db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners) != 1:
            raise ValueError('Unique provider ownership required')
        for name in provider['functions']:
            spec = specs[name]
            identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
            for table, key in [('atoms', 'atom_id'), ('artifacts', 'artifact_id')]:
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s', (spec.fqdn, identity)).fetchone():
                    raise ValueError('Provider identity already exists')
            for table in ('atom_versions', 'artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (spec.version_id,)).fetchone():
                    raise ValueError('Version identity collision')
            node = definitions[name]
            ports = [dict(direction='input', ordinal=i, name=a.arg, type_desc=ast.unparse(a.annotation),
                          required=True, default_value_repr='', constraints=CONSTRAINTS[a.arg])
                     for i, a in enumerate(node.args.args)]
            output = 'Ordered list of nonempty string tokens; empty input returns an empty list. Consecutive word or whitespace code points form runs; all other code points are individual tokens. Joining tokens exactly reproduces the input. Code-point semantics are not grapheme or linguistic segmentation. Independent chunk tokenization may change run boundaries. Witness metadata gives bounds only, not an exact token-array shape.'
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc=ast.unparse(node.returns),
                              required=True, default_value_repr='', constraints=output))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+name, atom_id=str(identity), version_id=str(spec.version_id),
                content_hash=spec.content_hash, provider_sha256=provider['sha256'], semver=spec.semver,
                fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='character-tokenizer-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records,
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=provider['revision'], planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Draft identities only; canonical draft and legacy flagged, both unpublished, until separate publication qualification.',
            'Witness describes token-count bounds and lossless code-point preservation only. Caller chooses domain-appropriate alphabets and preserves policy between training and inference.',
            'Requires the installed sciona tokenizer core and registry. Runs serially without worker processes. Dependency qualification remains pending.',
            'Complete metadata, dependency qualification, transaction rollback tests and served-selection verification remain required.',
            'No original competition CDG approval or historical winning-ensemble completeness is implied.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/character_tokenizer_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), ports_per_catalog=sum(len(p['interfaces']) for p in result['new_providers']), applied=False)))
