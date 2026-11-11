"""Read-only version-correction plan for the qualified combined graph providers."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from dotenv import dotenv_values
import psycopg
from psycopg.rows import dict_row

from sciona.atoms.supabase_seed import _version_material
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.qualify_osic_relocated_combined_graph import ROOT, stage_graphs

EVIDENCE_REVISION = 'f28f2dde'
EVIDENCE_PATH = 'docs/reviews/osic_relocated_combined_graph.json'


def material(repo, path, fqdn, symbol, source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol)
    decorators = [d for d in node.decorator_list if isinstance(d, ast.Call)
                  and isinstance(d.func, ast.Name) and d.func.id == 'register_atom']
    if len(decorators) != 1 or not isinstance(decorators[0].args[0], ast.Name):
        raise ValueError('Expected one named witness registration')
    witness_name = decorators[0].args[0].id
    witness = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == witness_name)
    version, content, semver, _, fingerprint = _version_material(
        repo_name=repo.name, file_path=path.relative_to(repo), fqdn=fqdn,
        source_symbol=symbol, witness_name=witness_name, source_text=source,
        node=node, witness=witness)
    return dict(version_id=version, content_hash=content, semver=semver, fingerprint=fingerprint)


def normalized(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == 'sciona._runtime.ml':
                node.module = 'sciona'
            elif node.module.startswith('sciona._runtime.ml.'):
                node.module = 'sciona.'+node.module[len('sciona._runtime.ml.'):]
    return ast.dump(tree, include_attributes=False)


def plan():
    raw = subprocess.check_output(['git','show',EVIDENCE_REVISION+':'+EVIDENCE_PATH], cwd=ROOT)
    if raw != (ROOT/EVIDENCE_PATH).read_bytes():
        raise ValueError('Immutable numerical relocation evidence changed')
    evidence = json.loads(raw)
    if not evidence['passed'] or evidence['approved'] or not evidence['synthetic_only']:
        raise ValueError('Expected successful unapproved synthetic evidence')
    def check_software():
        for name, expected in evidence['current_software_sha256'].items():
            if hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Numerically qualified software changed')
    check_software()
    with tempfile.TemporaryDirectory(prefix='sciona-osic-version-plan-') as temporary:
        stage_graphs(Path(temporary))
        sys.path.insert(0, temporary)
        try:
            from frozen_osic_graphs.osic_combined_inference_graph import build_combined_inference_graph
            graph = build_combined_inference_graph()
        finally:
            sys.path.remove(temporary)
    if encode_execution_graph(graph)[0] != evidence['execution']['graph_sha256']:
        raise ValueError('Qualified graph differs')
    repo = ROOT.parent/'sciona-atoms-ml'
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    records = []
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
                         row_factory=dict_row, options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        for target in sorted({n.matched_primitive for n in graph.nodes}):
            module, symbol = target.rsplit('.', 1)
            path = repo/'src'/Path(*module.split('.')).with_suffix('.py')
            source = path.read_text()
            committed = subprocess.check_output(['git','show',revision+':'+str(path.relative_to(repo))], cwd=repo).decode()
            if normalized(source) != normalized(committed):
                raise ValueError('Provider changes exceed runtime import relocation')
            rows = db.execute('''SELECT a.atom_id::text,a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                FROM atoms a JOIN atom_versions v USING(atom_id)
                WHERE a.import_module=%s AND a.source_symbol=%s AND v.is_latest''', (module,symbol)).fetchall()
            if len(rows) != 1:
                raise ValueError('Unique latest catalog provider required')
            old = rows[0]
            old_material = material(repo,path,old['fqdn'],symbol,committed)
            if old_material['version_id'] != old['version_id'] or old_material['content_hash'] != old['content_hash']:
                raise ValueError('Committed provider does not reproduce catalog identity')
            mirrored = db.execute('''SELECT v.version_id::text,v.content_hash,v.trust_tier
                FROM artifact_versions v WHERE v.artifact_id=%s AND v.is_latest''', (old['atom_id'],)).fetchall()
            if mirrored != [{k:old[k] for k in ('version_id','content_hash','trust_tier')}]:
                raise ValueError('Catalog versions disagree')
            for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
                if not db.execute(f'SELECT 1 FROM {table} WHERE {key}=%s',(old['atom_id'],)).fetchone():
                    raise ValueError('Expected prior approval in both served catalogs')
            current = material(repo,path,old['fqdn'],symbol,source)
            changed = current['content_hash'] != old['content_hash']
            records.append(dict(runtime_fqdn=target, artifact_id=old['atom_id'],
                previous_version_id=old['version_id'],previous_content_hash=old['content_hash'],
                previous_trust_tier=old['trust_tier'], proposed=current,
                provider_sha256=hashlib.sha256(source.encode()).hexdigest(),
                import_only_change_verified=True, current_source_committed=source==committed,
                action='new_provider_version_required' if changed else 'preserve_current_version'))
    check_software()
    changed = [r for r in records if r['action']=='new_provider_version_required']
    return dict(format='osic-relocated-provider-version-plan.v1',read_only=True,approved=False,
                catalog_mutations=0,graph_sha256=evidence['execution']['graph_sha256'],
                qualification_revision=EVIDENCE_REVISION,qualification_sha256=hashlib.sha256(raw).hexdigest(),
                previous_provider_revision=revision,providers=len(records),new_versions_required=len(changed),
                unchanged_versions=len(records)-len(changed),records=records,
                remaining_gates=[
                    'Commit the provider/runtime relocation with its owning work before anchoring new immutable provider revisions.',
                    'Review new versions and relocated transitive runtime identities against the existing contracts and current numerical evidence.',
                    'Validate draft intake, publication rollback and idempotency; retain previous versions and their evidence.',
                    'Resolve new versions through both served catalogs and rerun combined graph execution before graph publication.',
                    'Complete image training and the remaining original-workflow qualification separately.'])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/osic_relocated_provider_version_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
