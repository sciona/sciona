"""Freeze four image preparation/reduction stages and their reusable dependencies."""
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
MODULE='sciona.atoms.ml.domain_adapters.osic_image'
PROVIDER_PATH='src/sciona/atoms/ml/domain_adapters/osic_image.py'
PROVIDER_REVISION='ec50c40'
EVIDENCE_COMMIT='bddbcf34'
EVIDENCE='docs/reviews/osic_intermediate_composition.json'
TABLE='Nonempty pandas DataFrame with unique exact caller-mapped fields entity,time,target,numeric,category_a,category_b. Nonempty string entity identities; finite numeric coordinates. Caller owns coordinate units and actual observation availability. No source records or identifiers are embedded.'
IMAGE='Private nonempty encoded single-frame, single-sample monochrome DICOM bytes. Qualified provisioned pydicom/OpenCV runtime only. Uncompressed and RLE decoding have separate synthetic qualification; other compressed codecs remain unqualified. Caller disables OpenCV workers before execution.'
PORTS={
    'table':('pandas.DataFrame',TABLE+' Fit every entity before exclusions with explicit rcond=-1. Preserve first-encounter entity order and first-row metadata; minimum-norm slopes may be rank deficient.'),
    'encoded_planes':('dict[str,list[tuple[str,bytes]]]',IMAGE+' Exact coverage of all mapped entities, including excluded entities. Eligible entities require nonempty ordered uniquely named records and valid rescale fields with nonzero slope. Excluded payloads are not decoded.'),
    'excluded_entities':('list[str]','Explicit unique nonempty string exclusion identities. Exclusions affect image sampling eligibility after longitudinal fitting. At least one eligible entity must remain. Caller-owned policy, with no source exclusion list embedded.'),
    'time_coordinate':('str','Nonempty time-coordinate declaration; caller owns common units for fitting and later queries. No unit conversion.'),
    'response_coordinate':('str','Nonempty response-coordinate declaration. Fitted slopes have response units per time unit; labels do not prove compatible coordinates.'),
    'numeric':('number','Finite real scalar metadata coordinate, excluding bool. Apply source (value-30)/30 transform; no clinical or generic interpretation inferred.'),
    'category':('str','Source inference category string. Case-sensitive comparison to the source label; unknown strings retain the fallback. Not a generic categorical encoder.'),
    'status':('str','Source status string. Exact source cases and unknown-string fallback are retained. Availability and application interpretation remain caller responsibilities.'),
    'slice_slopes':('list[numpy.ndarray]','Nonempty ordered per-entity finite real arrays (S,) or (S,1), S>=2. Median retains input dtype behavior; do not promote float32 before reduction. Finite float64 representation required.'),
    'baseline_time':('numpy.ndarray','Finite real vector (G,) aligned with the slope groups. Same time coordinate as query times; converted to float64.'),
    'baseline_value':('numpy.ndarray','Finite real vector (G,) of already observed baseline outcomes aligned with slope groups. Response units must agree with slopes times time; converted to float64.'),
    'baseline_uncertainty':('numpy.ndarray','Finite real vector (G,) of caller-supplied source heuristic values, aligned with slope groups. Not calibrated uncertainty; negative values are not clipped.'),
    'query_subject':('numpy.ndarray','Integer vector (M,) indexing the supplied slope groups, with 0<=index<G. Explicit caller mapping; no entity joining occurs.'),
    'query_time':('numpy.ndarray','Nonempty finite real vector (M,) sharing baseline time units. Integer inputs retain source scalar-integer multiplication behavior; floating inputs use floating scalar multiplication. Finite float64 representation required.'),
    'image_keys':('list[str]','Nonempty unique nonempty string query keys aligned with image rows. Must have exactly the same set as quantile keys.'),
    'quantile_keys':('list[str]','Nonempty unique nonempty string query keys aligned with quantile rows. Must have exactly the same set as image keys.'),
    'image_values':('numpy.ndarray','Finite real matrix (M,2) of point and source uncertainty, aligned with image keys; finite float64 representation required. Caller establishes units and applicability.'),
    'quantile_values':('numpy.ndarray','Finite real matrix (M,2) of point and source uncertainty, aligned with quantile keys; finite float64 representation required. Same units as the image branch.'),
}
OUTPUTS={
    'prepare_image_training_population':('dict','Private entities, per-entity images (S,512,512,1), metadata (G,4), slopes (G,), fit diagnostics and coordinate declarations. Training normalization subtracts intercept, divides by slope*1000, then resizes. No model fitting or sampling; image groups are materialized for this training population.'),
    'prepare_image_slice_inputs':('dict','Private images (S,512,512,1), metadata (S,4) and int64 selected_positions. At least two selected slices; source strict ordinal/count bounds retain caller order. Decode selected payloads only, ignore rescale metadata, divide integer pixels by 2048 and resize. Only one entity is prepared; no model call.'),
    'reduce_image_predictions':('numpy.ndarray','Float64 (M,2) source point and heuristic uncertainty. Median each group; form the intercept before extrapolating. Preserve input-dtype median/scalar rounding and negative uncertainty. No calibrated interval claim.'),
    'blend_keyed_predictions':('tuple[list[str],numpy.ndarray]','Sorted keys and a fresh float64 (M,2) matrix. Align both branches by keys, then use ordered weighted sum with source weights 0.45 and 0.55. No fitting or empirical calibration.'),
}
DEPENDENCIES=[
    dict(artifact_id='c5a57e38-9d64-526e-8f6c-953fb10545f1',fqdn='sciona.atoms.numerical.statistics.affine_fit.affine_least_squares',version_id='a9cae503-875f-5dec-b82b-a113c8aec6cf',content_hash='115433af2fa42aa4454253a18682acef7c1447142261590539cbef9da6da5896',trust_tier=3),
    dict(artifact_id='4f486f1c-82db-50db-b5d2-188aeaa8cc4a',fqdn='sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum',version_id='5fb9c91f-ff5f-590c-b376-88e2a7360440',content_hash='814d0ff20f33820537741e1f116c32dfa69262d56dc16d872f3a055f77bd6968',trust_tier=3),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    repo=ROOT.parent/'sciona-atoms-ml'
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+EVIDENCE],cwd=ROOT)
    report=json.loads(raw)
    execution=report['execution']
    if (raw!=(ROOT/EVIDENCE).read_bytes() or report.get('passed') is not True
            or report.get('synthetic_only') is not True
            or set(execution['provider_interfaces_executed'])!=set(OUTPUTS)
            or not all(execution.get(k) is True for k in ('training_arrays_and_rng_exact',
                'source_inference_exact','separate_image_preparation_and_reduction_exact',
                'numeric_state_restored','fitting_disabled','private_artifacts_unchanged'))
            or execution['source_training_samples']!=24 or execution['actual_b5_inference_queries']!=4
            or execution['composition']['exact_blend'] is not True):
        raise ValueError('Exact passing connected execution evidence required')
    committed=subprocess.check_output(['git','show',PROVIDER_REVISION+':'+PROVIDER_PATH],cwd=repo)
    if committed!=(repo/PROVIDER_PATH).read_bytes() or sha(repo/PROVIDER_PATH)!=report['image_provider_sha256']:
        raise ValueError('Immutable provider changed')
    implementations=report['software_sha256']
    for name,digest in implementations.items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified implementation changed: '+name)
    dependency_implementations={
        'sciona-atoms/src/sciona/atoms/numerical/statistics/affine_fit.py':report['affine_provider_sha256'],
        'sciona-atoms/src/sciona/atoms/numerical/statistics/ordered_sum.py':report['numeric_provider_sha256']['ordered_weighted_sum'],
    }
    for name,digest in dependency_implementations.items():
        if sha(ROOT.parent/name)!=digest:raise ValueError('Qualified generic provider changed')
    expected_dependencies=sorted(DEPENDENCIES,key=lambda d:d['fqdn'])
    source_specs=[s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name,repo),
        artifact_root=repo/'src/sciona/atoms') if s.import_module==MODULE]
    if len(source_specs)!=4 or {s.source_symbol for s in source_specs}!=set(OUTPUTS):
        raise ValueError('Exactly four image stages required')
    definitions={n.name:n for n in ast.parse(committed).body if isinstance(n,ast.FunctionDef)}
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        dependencies=expected_dependencies
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
            if spec.source_symbol=='prepare_image_slice_inputs':
                port=next(p for p in ports if p['name']=='encoded_planes')
                port.update(type_desc='list[tuple[str,bytes]]',constraints=IMAGE+
                    ' Ordered uniquely named records for exactly one entity. Source names strip four trailing characters to parse integer slice ordinals. Rescale metadata is ignored; unselected payloads are not decoded.')
            output_type,output=OUTPUTS[spec.source_symbol]
            ports.append(dict(direction='output',ordinal=0,name='result',type_desc=output_type,required=True,
                default_value_repr='',constraints=output+' Concrete provisioned execution only; abstract witness rejects.'))
            fields={k:getattr(spec,k) for k in ('fqdn','namespace_root','namespace_path','repo_name','source_module_path',
                'import_module','source_symbol','description','domain_tags','source_kind','is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol,atom_id=str(identity),version_id=str(spec.version_id),
                content_hash=spec.content_hash,provider_sha256=sha(repo/PROVIDER_PATH),semver=spec.semver,
                fingerprint=comparable(spec.fingerprint),seed_fields=comparable(fields),interfaces=ports,
                registered_provider_dependencies=([dependencies[0]] if spec.source_symbol=='prepare_image_training_population' else
                    [dependencies[1]] if spec.source_symbol=='blend_keyed_predictions' else [])))
    return dict(format='osic-image-intake-plan.v1',passed=True,applied=False,approved=False,catalog_mutations=0,
        owner=owners[0],new_providers=records,registered_provider_dependencies=dependencies,
        execution_evidence_commit=EVIDENCE_COMMIT,execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION,implementation_sha256=implementations,planner_sha256=sha(__file__),
        dependency_implementation_sha256=dependency_implementations,
        limitations=['Draft plan only; transaction, semantic, runtime and publication gates remain required.',
            'Four source-specific preparation/reduction interfaces only. Generic affine fitting and ordered summation remain independent approved atoms.',
            'Training and inference use different source image normalization and category policies. Caller mapping and observation availability are external.',
            'The execution oracle uses a private intermediate synthetic image checkpoint, not completion of the 30-epoch image training schedule.',
            'Thin interfaces depend on separately pinned matcher implementations and provisioned decoder/model dependencies; provider source hashes alone do not qualify behavior.',
            'No image-model training provider, original full workflow or final catalog graph is approved.'])



if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/osic_image_intake_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']),dependencies=len(result['registered_provider_dependencies']),applied=False)))
