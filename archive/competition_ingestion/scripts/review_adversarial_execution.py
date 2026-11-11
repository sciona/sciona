"""Reproducible automated Tier3 source/computation/evidence audit. No DB writes."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
EVIDENCE=('backbone_review','ensemble','environment','graph_contract','graph_execution','image_boundary',
          'inception_ops','inception_resnet','inception_v3','inception_v4','json_execution','lifecycle',
          'losses','normalization','padded_attack','pooling_reference','resnet','resnet_spatial',
          'runtime_boundary','scipy_reference','source_pins','source_review','state_mapping',
          'tensorflow_reference','updates')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit():
    reviews=ROOT/'docs/reviews';documents={}
    for suffix in EVIDENCE:
        p=reviews/('competition_adversarial_'+suffix+'.json')
        document=json.loads(p.read_text());documents[suffix]=document
        for name,digest in document.get('sha256',{}).items():
            target=(ROOT/name).resolve()
            assert target.is_relative_to(ROOT) and target.is_file() and sha(target)==digest,name
    for suffix in set(EVIDENCE)-{'backbone_review','environment','pooling_reference','scipy_reference','source_pins','source_review','tensorflow_reference'}:
        assert documents[suffix]['result']=='passed',suffix
    pins=documents['source_pins']
    assert pins['intake_version']=='cebbfc7d-e32e-5f52-9ec2-e1de0d955501'
    assert pins['intake_content_hash']=='d6b977ae079c82dec43ffdf88f64fb9089c17e587c3ee4c36f5411aa0b28828b'
    assert {s['branch']:s['commit'] for s in pins['sources']}=={
        'non_targeted':'5c68162e05de7afed2e3d33115df43e2a7a1c3da',
        'targeted':'7da707fe568053a9e2735fcc692c8f0ad7e122a1'}
    for source in pins['sources']:
        assert sha(ROOT/source['license_notice'])==source['license_sha256']
        for item in source['files']:
            p=Path('/private/tmp/sciona_adversarial_source')/source['branch']/item['path']
            data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_sha1']
    for suffix,cache,license_name in [
        ('tensorflow_reference','sciona_adversarial_tensorflow_source','Adversarial-TensorFlow-reference-Apache-2.0.txt'),
        ('pooling_reference','sciona_adversarial_tensorflow_source',None),
        ('scipy_reference','sciona_adversarial_scipy_source','Adversarial-SciPy-reference-BSD.txt')]:
        for item in documents[suffix]['files']:
            assert sha(Path('/private/tmp')/cache/item['path'])==item['sha256']
            if license_name and item['path'] in ('LICENSE','LICENSE.txt'):
                assert sha(ROOT/'docs/licenses'/license_name)==item['sha256']
    assert documents['updates']['checks']['source_tail_cases']==32
    assert documents['updates']['checks']['source_nan_and_explicit_rejection_cases']==3
    assert documents['losses']['checks']['head_gradients_compared']==58
    assert documents['losses']['checks']['probability_consensus_not_majority']
    for branch,models,steps in [('non_targeted',8,10),('targeted_large',5,20),('targeted_small',2,40)]:
        assert documents['ensemble']['checks'][branch]['full_models']==models
        assert documents['lifecycle']['checks'][branch]['complete_iterations']==steps
        assert documents['lifecycle']['checks'][branch]['independent_momentum_checks']==steps
    assert documents['padded_attack']['checks']['full_neural_batch_size']==10
    assert documents['padded_attack']['checks']['targeted_small_full_iterations']==40
    assert documents['json_execution']['checks']['actual_full_models']==8
    assert documents['json_execution']['checks']['neural_batch_size']==10
    full=documents['graph_execution']
    assert full['checks']=={'actual_runner_nodes':2,'actual_full_models':5,'complete_targeted_large_steps':20,
                           'neural_batch_size':10,'real_entries':1,'padded_entries':9,'encoded_caller_states':True,
                           'normalized_projection_bound':True,'only_real_source_scaled_PNG':True,'post_discovery_rng_unchanged':True}
    assert documents['image_boundary']['checks']['source_minmax_stretch_and_constant_black']
    runtime_files={str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('adversarial_*.py')}
    runtime_files|={str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('adversarial_*topology.json')}
    assert runtime_files <= set(full['sha256'])
    provider=ROOT.parent/'sciona-atoms-dl/src/sciona/atoms/dl/adversarial_execution.py'
    assert sha(provider)==full['provider_sha256']==documents['graph_contract']['provider_sha256']
    from sciona.adversarial_graph import build_adversarial_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_adversarial_graph())[0]==full['serialized_graph_sha256']
    dependencies={}
    for line in (ROOT/'requirements/adversarial-execution.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name,version=line.split('==');assert importlib.metadata.version(name)==version
            dependencies[name]=version
    assert dependencies==documents['environment']['direct_runtime_versions']
    stages=[
        ('ensemble_prediction_label_inference','Unweighted eight-model probabilities on iteration0 then fixed labels',['losses','ensemble','lifecycle']),
        ('adaptive_epsilon_attack_strategy','Separate targeted epsilon>=8 five-model20steps versus epsilon<8 two-model40steps; non-targeted eight-model10step default',['updates','lifecycle','padded_attack','graph_execution']),
        ('ensemble_logit_fusion_with_asymmetric_weights','Source-ordered main and auxiliary weighted logits across actual full backbones',['losses','ensemble']),
        ('auxiliary_logit_loss_fusion','Inference auxiliary heads retained; main CE plus0.4 auxiliary CE',['inception_v3','inception_v4','inception_resnet','losses','graph_execution']),
        ('momentum_iterative_gradient_accumulation','Actual neural input gradients,mean-absolute non-targeted normalization and recurrent momentum',['updates','ensemble','lifecycle']),
        ('std_normalized_momentum_gradient','Targeted population-std input-gradient normalization and post-accumulation renormalization',['updates','lifecycle']),
        ('rounded_clipped_perturbation_step','Targeted half-even rounding,step clipping and fixedoriginalprojection;non-targeted sign ascent',['updates','lifecycle','image_boundary'])]
    limitations=[
        'Automated Tier3 Community execution reconstruction; no Tier1 human certification or Tier2 usage qualification. Original intake remains draft.',
        'Intake cites the momentum paper and detailed algorithms but contains no explicit repository reference field. Author repository correspondence is an assessed provenance link, not an invented version-bound catalog URL.',
        'Full source-shaped299RGB1001class backbones and auxiliary heads execute. CPU Torch/NCHW operations and initializers adapt unspecified historical TF binaries; numerical reduction and max-pool tie-gradient parity unverified.',
        'Exact caller model states or explicit random initialization required. Synthetic tests do not establish pretrained attack effectiveness or historical checkpoint authenticity. Named-tensor conversion is not TensorFlow checkpoint binary decoding.',
        'Undefined zero gradient/variance normalization explicitly rejects instead of source NaNs; no stabilizing epsilon. Synthetic padded targeted evidence uses supplied nonzero BN offsets, not a silent initialization change.',
        'Source ten-entry zero padding,zero target padding,and real-entry trimming preserved. Evidence executes full10/20/40schedules and padded batches in all branches; real graph runner evidence covers five-model targeted-large.',
        'Source SciPy reference imsave stretches per-image min/max and maps constantfloatimages to black. Only normalized outputs have epsilon projection guarantee; saved PNG pixel perturbation bound is NOT guaranteed.',
        'TensorFlow1.4/SciPy0.19.1 are explicit reference versions, not asserted competition versions. PNG decoded pixels checked with current Pillow; historical bytes unverified.',
        'Sequential per-model VJP recomputation preserves ensemble objective but can change floating summation order. Direct full two-model autograd comparison passed; historical TF whole-ensemble parity unverified.',
        'Runtime arrays,states and PNGs are private caller data; only synthetic aggregate evidence retained. No downloads or payload logging. Graph validation captures intermediate persistence only in memory.',
        'Installed numerical dependencies pinned; clean installation untested. Matcher supplies broader runner dependencies. RNG checks start after provider discovery; unrelated cold imports may affect NumPy state.'
    ]
    auxiliary=['requirements/adversarial-execution.txt','scripts/review_adversarial_execution.py']
    auxiliary += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'docs/licenses').glob('Adversarial-*'))]
    return {'format':'adversarial-semantic-review.v1','review_source':'automated','proposed_tier':3,
            'verdict':'acceptable_with_limits','source_version_id':pins['intake_version'],'source_hash':pins['intake_content_hash'],
            'source_commits':{s['branch']:s['commit'] for s in pins['sources']},
            'serialized_graph_sha256':full['serialized_graph_sha256'],'provider_sha256':sha(provider),
            'provider_package_sha256':sha(ROOT.parent/'sciona-atoms-dl/pyproject.toml'),
            'intake_stage_mapping':[{'intake':s,'realization':r,'evidence':e} for s,r,e in stages],
            'limitations':limitations,'dependencies':dependencies,
            'finding':'Complete source-specific computational closure and actual serialized graph evidence support Tier3 selection with explicit initialization and stated numerical/output limits. Initial implementation-required reviews remain historical and are superseded for computational readiness. Catalog mutation and served verification gates remain separate.',
            'evidence_sha256':{'competition_adversarial_'+s+'.json':sha(reviews/('competition_adversarial_'+s+'.json')) for s in EVIDENCE},
            'auxiliary_sha256':{p:sha(ROOT/p) for p in auxiliary},'catalog_mutations':0}


def main():
    result=audit()
    (ROOT/'docs/reviews/competition_adversarial_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verdict':result['verdict'],'proposed_tier':3,'intake_stages':len(result['intake_stage_mapping']),
                      'evidence_documents':len(result['evidence_sha256']),'catalog_mutations':0}))


if __name__=='__main__':main()
