"""Review immutable original stages against the full sharded realization."""
import ast
import json
import subprocess
from scripts.audit_decomposed_video_multilabel_readiness import audit, ROOT, sha
from scripts.import_flavours_provider_drafts import connect
from sciona.decomposed_video_multilabel_graph import build_decomposed_video_multilabel_graph

STAGES={
    'feature_stream_loading':dict(
        anchors=[('video_multilabel_shards','FeatureShards'),('video_multilabel_shards','validate_populations'),('video_multilabel_sharded_training','fit')],
        realization='Read indexed local shards on demand; preserve all configured batches and epochs and fit sparse vocabulary only from training rows.',
        limits='Caller constructs immutable FeatureShards stores outside the graph from private runtime configuration. Record reads are bounded, but index metadata, labels, vocabulary, sparse training CSR, fitted state and scores remain resident. No constant total memory or distributed execution claim.'),
    'temporal_pooling':dict(
        anchors=[('video_multilabel_network','VideoHead.__init__'),('video_multilabel_reusable_network','ReusableVideoHead.pool')],
        realization='Select the allowed learned attention branch, with masked softmax pooling exposed as a reusable differentiable atom.',
        limits='Precomputed frame vectors only. No raw feature extraction or temporal position encoding; this selected pooling is invariant to frame order. Other listed alternative families are not claimed.'),
    'context_gated_multilabel_head':dict(
        anchors=[('video_multilabel_reusable_network','ReusableVideoHead.forward'),('video_multilabel_sharded_training','fit')],
        realization='Learn attention, context affine gates and per-label logits with CPU float64 Adam/BCE for every configured epoch. Generic sigmoid gating remains independently selectable.',
        limits='Architecture dimensions, optimizer settings and independent output labels are recipe choices. No hierarchy or label-embedding branch is claimed.'),
    'sparse_baseline_fusion':dict(
        anchors=[('video_multilabel_sharded_training','fit'),('video_multilabel_sharded_training','ShardedVideoEnsemble.components'),('video_multilabel_sharded_training','ShardedVideoEnsemble.predict')],
        realization='Fit one sparse L2 logistic model per label with training-only vocabulary; fuse neural and sparse probabilities by equal columnwise mean using the existing approved provider.',
        limits='Rare-label coverage improvement is an intended modeling benefit, not established by synthetic parity. Both training classes are required; unknown query features are ignored. No accuracy guarantee.'),
    'rank_ensemble_calibration':dict(
        anchors=[('video_multilabel_sharded_training','execute')],
        realization='Select the allowed label-specific threshold branch: maximize empirical F1 on disjoint held-out calibration groups, retain highest threshold on ties, and apply inclusive thresholds to every query row.',
        limits='Calibration requires both classes per label. No rank averaging, prior correction, top-k format, calibrated-probability or unbiased-performance claim.'),
}


def anchor(module,symbol):
    path=ROOT/'sciona'/f'{module}.py';node=ast.parse(path.read_text())
    for part in symbol.split('.'):
        matches=[n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==part]
        if len(matches)!=1:raise ValueError('Missing source anchor')
        node=matches[0]
    return dict(file=str(path.relative_to(ROOT)),symbol=symbol,line=node.lineno,file_sha256=sha(path))


def review():
    state=audit()
    if state['implementation_drift']:raise ValueError('Retained reference changed')
    if set(STAGES)!={s['stage_id'] for s in state['source_stages']}:raise ValueError('Every original stage must be reviewed')
    evidence='docs/reviews/decomposed_video_multilabel_graph_execution_v4.json'
    raw=(ROOT/evidence).read_bytes()
    if raw!=subprocess.check_output(['git','show','8673d9be:'+evidence],cwd=ROOT):raise ValueError('Immutable execution evidence changed')
    execution=json.loads(raw)
    for path,digest in execution['software_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Executed software changed')
    mappings=[dict(source_stage=source,**{k:v for k,v in STAGES[source['stage_id']].items() if k!='anchors'},
        source_anchors=[anchor(*item) for item in STAGES[source['stage_id']]['anchors']]) for source in state['source_stages']]
    providers=[];differences=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for node in build_decomposed_video_multilabel_graph().nodes:
            if node.node_id=='execute':continue
            rows=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier,
                EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
                FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest''',(node.matched_primitive,)).fetchall()
            if len(rows)!=1 or rows[0]['status']!='approved' or not rows[0]['is_publishable'] or not rows[0]['served'] or rows[0]['trust_tier']!=3:
                raise ValueError('Exact approved Tier 3 numeric identity required')
            record=rows[0]
            ports=db.execute('''SELECT direction,name,ordinal,type_desc FROM artifact_io_specs
                WHERE version_id=%s ORDER BY direction,ordinal''',(record['version_id'],)).fetchall()
            expected=[dict(direction=d,name=p.name,ordinal=i,type_desc=p.type_desc)
                for d,items in [('input',node.inputs),('output',node.outputs)] for i,p in enumerate(items)]
            if ports!=expected:differences.append(dict(node_id=node.node_id,catalog_ports=ports,draft_ports=expected))
            providers.append(record)
    return dict(format='video-original-scope.v1',read_only=True,catalog_mutations=0,approved=False,
        original=state['original'],source_version_id=state['source_version_id'],source_stage_review=mappings,
        numeric_providers=providers,graph_interface_differences=differences,
        execution_evidence_sha256=sha(ROOT/evidence),reviewer_sha256=sha(__file__),
        remaining_requirements=['Qualify and publish the recipe boundary with exact approved callable dependencies.',
            'Construct the corrected graph using exact catalog output names/types, including the reused mean provider, and rerun stored native execution.',
            'Stage a corrected immutable original version with complete input contracts; validate rollback, eligibility and served selection before approval.'],
        limits=['Source alternatives are explicit selected branches, not implementations of every listed model family.',
                'This scope review does not approve the original CDG or the recipe boundary.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/video_original_scope.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(stages=len(result['source_stage_review']),approved_numeric_providers=len(result['numeric_providers']),
        graph_interface_differences=len(result['graph_interface_differences']),approved=False)))
