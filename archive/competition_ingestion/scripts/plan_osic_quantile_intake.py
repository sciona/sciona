"""Freeze five source-specific quantile stages and exact loss dependencies."""
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
from scripts.plan_group_distribution_intake import check_dependency

ROOT=Path(__file__).resolve().parents[1]
MODULE='sciona.atoms.ml.domain_adapters.osic_quantile'
PROVIDER_PATH='src/sciona/atoms/ml/domain_adapters/osic_quantile.py'
PROVIDER_REVISION='0b7695b'
EVIDENCE_COMMIT='59c7e284'
EVIDENCE='docs/reviews/osic_training_boundary.json'
TABLE='Nonempty complete pandas DataFrame with exact caller-mapped fields entity,time,target,numeric,category_a,category_b. String labels must be nonblank; numeric coordinates finite. Caller owns units and observation availability. No real records or metadata are embedded in this contract.'
PORTS={
    'training':('pandas.DataFrame',TABLE+' All duplicate entity/time rows are removed, not collapsed to one row; encounter order is retained.'),
    'baseline':('pandas.DataFrame',TABLE+' Exactly one supplied baseline per queried entity; supplied outcomes must already be observable.'),
    'queries':('pandas.DataFrame','Nonempty exact entity,time fields with complete finite numeric times and nonblank entity strings. Every entity must have a supplied baseline. Query order and the complete population are significant.'),
    'population':('pandas.DataFrame',TABLE+' Additional population field must contain exactly training,baseline,query roles. Produced by assemble_quantile_populations; retain row order.'),
    'allow_transductive':('bool','Must be exactly True. Authorizes fitting joint categories and ranges using the query population; changed queries can alter training features.'),
    'features':('numpy.ndarray','Finite float32 matrix (N,D), D>=1. Training requires N>=2; replay requires N>=1. Caller aligns columns to explicit feature coordinates and owns prior preparation.'),
    'targets':('numpy.ndarray','Finite float32 matrix (N,1), row-aligned with training features, in the declared response units. No response scaling is inferred.'),
    'query_features':('numpy.ndarray','Finite float32 matrix (M,D), M>=1, with the same ordered feature meanings and units as training.'),
    'columns':('list[str]','D unique nonblank ordered feature names. Must agree exactly with saved state on replay; names do not prove semantic equivalence.'),
    'feature_coordinate':('str','Nonblank feature-space declaration matching saved states exactly. Caller owns feature construction and alignment.'),
    'response_coordinate':('str','Nonblank response-space declaration matching saved states exactly. Caller owns target units; no inverse transform is inferred.'),
    'seeds':('tuple[int,int]','Exactly two Python integers in [0,2**31), excluding bool. Resets Keras/random seeds for each fold; historical random draws are not reproduced. Serial execution only.'),
    'states':('list[dict]','Exactly two private numeric sciona.osic_quantile_prediction.v1 states with fixed eight-array float32 weight inventory and matching columns/coordinates. No executable graph or optimizer state; caller verifies authenticity.'),
    'folds':('numpy.ndarray','Nonempty finite float32 or float64 (F,M,3) outputs in aligned query order. Divide each fold in its original dtype before float64 accumulation. Columns are low,point,high; crossings are retained.'),
    'observed_mask':('numpy.ndarray','Boolean vector (M,) marking only outcomes already known at prediction time. Caller establishes availability, not merely event time.'),
    'observed_values':('numpy.ndarray','Finite real vector (M,) in response units. Only masked rows are used, but all rows must be finite.'),
}
OUTPUTS={
    'assemble_quantile_populations':('pandas.DataFrame','Joint ordered training/baseline/query table with population roles; training duplicates removed and query metadata joined. No feature fitting.'),
    'fit_joint_quantile_features':('dict','Per-role feature matrices, targets and entity/time order plus ordered columns, categories and ranges. Source encounter-order category encoding and min/max scaling; reject degenerate ranges and cross-field category-name collisions. Must cast model inputs explicitly to float32. This is joint transductive preparation, not a train-only scaler.'),
    'train_prepared_quantile_folds':('dict','Raw fold_predictions (2,M,3), oof_predictions (N,3), two private fold_states and aggregate fold records. Fixed source architecture, two contiguous folds, 600 epochs, batches of 128 including partial batches, legacy Adam and source mixed loss. No fold reduction or baseline override. Model state is inference-only.'),
    'replay_prepared_quantile_folds':('numpy.ndarray','Float32 raw predictions (2,M,3) restored from both states. No preparation or fitting. Query features must match the original preparation scope; no baseline policy.'),
    'reduce_quantile_predictions':('numpy.ndarray','Float64 (M,2) point and raw uncertainty. Average folds in source order; midpoint is column 1 and spread is high-low. Preserve negative spreads. Override masked points with observed values and uncertainty 0.1 before branch blend. This is a source policy, not calibrated uncertainty.'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    repo=ROOT.parent/'sciona-atoms-ml'
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+EVIDENCE],cwd=ROOT)
    report=json.loads(raw)
    if raw!=(ROOT/EVIDENCE).read_bytes() or not all(report.get(k) is True for k in (
            'passed','synthetic_only','lifecycle_outputs_exact','private_fold_states_exact',
            'numeric_boundary_without_preparation','raw_fold_replay_exact','replay_training_and_preparation_disabled')):
        raise ValueError('Exact passing connected execution evidence required')
    if report['full_schedule_fits']!=6 or report['epochs_per_fit']!=600 or len(report['rejected_before_model_construction'])!=8:
        raise ValueError('Complete boundary qualification required')
    committed=subprocess.check_output(['git','show',PROVIDER_REVISION+':'+PROVIDER_PATH],cwd=repo)
    if committed!=(repo/PROVIDER_PATH).read_bytes() or sha(repo/PROVIDER_PATH)!=report['provider_sha256']:
        raise ValueError('Immutable provider changed')
    if (sha(ROOT/'sciona/osic_quantile_training.py')!=report['training_sha256']
            or sha(ROOT/'scripts/validate_osic_training_boundary.py')!=report['validator_sha256']):
        raise ValueError('Qualified lifecycle changed')
    # Bind every local implementation delegated to by these thin interfaces.
    implementation_paths=['sciona/osic_quantile_features.py','sciona/osic_quantile_training.py',
        'sciona/osic_quantile_model.py','sciona/osic_quantile_objective.py','sciona/osic_quantile_state.py',
        'sciona/osic_predictions.py','sciona/tensorflow_distribution_losses.py']
    implementations={p:sha(ROOT/p) for p in implementation_paths}
    loss_review_raw=subprocess.check_output(['git','show',
        '43c24f6b:docs/reviews/distribution_loss_semantics.json'],cwd=ROOT)
    loss_review=json.loads(loss_review_raw)
    if sha(repo/'src/sciona/atoms/ml/tensorflow_distribution_losses.py')!=loss_review['software_sha256']['provider']:
        raise ValueError('Qualified loss implementation changed')
    expected_dependencies=sorted([dict(artifact_id=p['artifact_id'],fqdn=p['fqdn'],
        version_id=p['version_id'],content_hash=p['content_hash'],trust_tier=3)
        for p in loss_review['providers']],key=lambda p:p['fqdn'])
    source_specs=[s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name,repo),
        artifact_root=repo/'src/sciona/atoms') if s.import_module==MODULE]
    if len(source_specs)!=5 or {s.source_symbol for s in source_specs}!=set(OUTPUTS):
        raise ValueError('Exactly five quantile stages required')
    definitions={n.name:n for n in ast.parse(committed).body if isinstance(n,ast.FunctionDef)}
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        dependencies=db.execute("""SELECT a.artifact_id::text,a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn=ANY(%s) AND v.is_latest ORDER BY a.fqdn""",([
                'sciona.atoms.ml.tensorflow_distribution_losses.mean_pinball',
                'sciona.atoms.ml.tensorflow_distribution_losses.clipped_laplace_score'],)).fetchall()
        if dependencies!=expected_dependencies: raise ValueError('Exact qualified loss versions required')
        for dependency in dependencies: check_dependency(db,dependency)
        owners=db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners)!=1: raise ValueError('Unique provider ownership required')
        for spec in sorted(source_specs,key=lambda s:s.source_symbol):
            identity=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)
            for table,key in [('atoms','atom_id'),('artifacts','artifact_id')]:
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s',(spec.fqdn,identity)).fetchone():
                    raise ValueError('Provider identity already exists')
            for table in ('atom_versions','artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s',(spec.version_id,)).fetchone():
                    raise ValueError('Version identity collision')
            ports=[dict(direction='input',ordinal=i,name=a.arg,type_desc=PORTS[a.arg][0],required=True,
                default_value_repr='',constraints=PORTS[a.arg][1]) for i,a in enumerate(definitions[spec.source_symbol].args.args)]
            output_type,output=OUTPUTS[spec.source_symbol]
            ports.append(dict(direction='output',ordinal=0,name='result',type_desc=output_type,required=True,
                default_value_repr='',constraints=output+' Concrete provisioned execution only; abstract witness rejects.'))
            fields={k:getattr(spec,k) for k in ('fqdn','namespace_root','namespace_path','repo_name','source_module_path',
                'import_module','source_symbol','description','domain_tags','source_kind','is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol,atom_id=str(identity),version_id=str(spec.version_id),
                content_hash=spec.content_hash,provider_sha256=sha(repo/PROVIDER_PATH),semver=spec.semver,
                fingerprint=comparable(spec.fingerprint),seed_fields=comparable(fields),interfaces=ports,
                registered_provider_dependencies=dependencies if spec.source_symbol in (
                    'train_prepared_quantile_folds','replay_prepared_quantile_folds') else []))
    return dict(format='osic-quantile-intake-plan.v1',passed=True,applied=False,approved=False,catalog_mutations=0,
        owner=owners[0],new_providers=records,registered_provider_dependencies=dependencies,
        execution_evidence_commit=EVIDENCE_COMMIT,execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION,implementation_sha256=implementations,planner_sha256=sha(__file__),
        dependency_review_commit='43c24f6b',dependency_review_sha256=hashlib.sha256(loss_review_raw).hexdigest(),
        limitations=['Draft plan only; transaction, semantic, runtime and publication gates remain required.',
            'Replay reconstructs the compiled fixed model and therefore retains the registered loss dependencies even though no optimizer update occurs.',
            'Thin interfaces depend on the separately pinned matcher implementations; provider source hashes alone do not qualify their behavior.',
            'Original full OSIC workflow, image branch and final catalog graph remain unapproved.'])


if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/osic_quantile_intake_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']),dependencies=len(result['registered_provider_dependencies']),applied=False)))
