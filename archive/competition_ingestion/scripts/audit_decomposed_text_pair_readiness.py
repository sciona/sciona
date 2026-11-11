"""Audit original text-pair scope, existing recipe and reuse without approval."""
import hashlib
import importlib.metadata as metadata
import json
from sciona.competition_import import _canonical
from scripts.import_flavours_provider_drafts import ROOT, connect

SOURCE='1169253d-76d3-5b76-885f-07b58bed795c'
SOURCE_HASH='896ea8e61c63f03fcb22b6bc418126ac23acf0fb3ad55e5a11c3815ae00e1bd7'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    execution=json.loads((ROOT/'docs/reviews/competition_text_pair_graph_execution.json').read_text())
    semantic=json.loads((ROOT/'docs/reviews/competition_text_pair_semantic_review.json').read_text())
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,e.details
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            JOIN artifact_audit_evidence e USING(version_id)
            WHERE v.version_id=%s AND e.runner_version='competition-intake.v1' ''',(SOURCE,)).fetchall()
        if len(rows)!=1 or rows[0]['content_hash']!=SOURCE_HASH:raise ValueError('Exact source required')
        original=dict(rows[0]); snapshot=original.pop('details')['snapshot']
        if hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=SOURCE_HASH:
            raise ValueError('Original source snapshot changed')
        derivative=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,
            EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn=%s AND v.is_latest''',(original['fqdn']+'.execution',)).fetchall()
        if len(derivative)!=1 or derivative[0]['content_hash']!=execution['serialized_graph_sha256']:
            raise ValueError('Retained derivative differs')
        candidates=db.execute('''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,
            v.trust_tier,artifact_is_publishable(a.artifact_id) AS native_eligible,
            EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.artifact_kind='atom' AND v.is_latest AND a.fqdn ILIKE ANY(%s) ORDER BY a.fqdn''',
            (['%jaccard%','%levenshtein%','%cosine%','%tfidf%','%truncated_svd%',
              '%empirical_f1%','%materialized_prediction.average'],)).fetchall()
    return dict(format='decomposed-text-pair-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        original=original,source_version_id=SOURCE,
        source_stages=[dict(stage_id=s['stage_id'],description=s['description'],guarantees=s['guarantees'])
                       for s in snapshot['template']['stages']],
        reference=derivative[0],reuse_candidates=candidates,
        implementation_drift=[p for p,d in execution['code_sha256'].items() if sha(ROOT/p)!=d],
        runtime_drift={n:dict(qualified=v,current=metadata.version(n)) for n,v in semantic['dependencies'].items()
                       if metadata.version(n)!=v},
        inspected_software_sha256={p:sha(ROOT/p) for p in [
            'sciona/text_pair_features.py','sciona/text_pair_training.py','sciona/text_pair_contract.py',
            'sciona/text_pair_graph.py','scripts/review_text_pair_execution.py',
            'docs/reviews/competition_text_pair_graph_execution.json','docs/reviews/competition_text_pair_semantic_review.json']},
        findings=[
            'Original pair_normalization names spelling correction, synonym/query expansion and domain normalization. Existing NFKC/casefold/word-token processing omits these; derivative limitations do not authorize removing original requirements.',
            'Original lexical_similarity_features names BM25 alongside overlap, TF-IDF, edit, exact-match and length features. Existing twelve-feature recipe has no BM25 computation. Resolve original scope through implementation or explicit source-supported alternative semantics before promotion.',
            'LSA/SVD, candidate graph statistics and logistic/forest ensemble are explicitly listed alternatives. Training-only fits and current-edge exclusion must remain intact.',
            'Original metric_thresholding promises calibrated submission outputs. Existing binary F1 threshold and JSON result need an explicit metric/output realization review; empirical F1 selection is not probability calibration.',
            'Existing recipe embeds generic edit distance, set Jaccard, cosine/distance, arithmetic averaging and threshold fitting. Inspect exact existing providers before introducing independently selectable replacements.',
            'The approved ordered mean and empirical-F1 providers are reuse candidates, not yet qualified substitutions: preserve exact floating arithmetic, tie rule, shape conversion and reported calibration F1.',
            'Name-pattern catalog inventory is a starting point, not proof that equivalent generic providers do not exist. Broaden semantic/source inventory before adding new atoms.',
            'Text normalization, vocabulary/model choices, feature assembly and population separation remain explicit recipe policy; generic numeric contracts must not hard-code text semantics.'],
        next_actions=[
            'Qualify current runtime with existing full synthetic feature, training and strict-boundary tests without rewriting historical approval evidence.',
            'Resolve missing original normalization/BM25 and output requirements before claiming full source coverage.',
            'Inventory and qualify reusable operations, then connect exact approved bindings in a complete original graph.',
            'Execute the stored candidate and validate publication rollback, native eligibility and served selection.'],
        limits=['Read-only scope and reuse audit; no new execution or promotion claim. Software and generic algorithm-template metadata only.'],
        auditor_sha256=sha(ROOT/'scripts/audit_decomposed_text_pair_readiness.py'))


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/decomposed_text_pair_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(original=result['original']['status'],reference_served=result['reference']['served'],
        implementation_drift=result['implementation_drift'],runtime_drift=result['runtime_drift'],
        reuse_candidates=len(result['reuse_candidates']),approved=False)))
