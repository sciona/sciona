"""Compare synthetic batch assembly to the pinned source sampling method."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciona.group_member_sampling import sample_indices
from sciona.osic_image_batches import sample_batch
from scripts.review_osic_original_scope import TRAIN, REPOSITORY, REVISION

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PIN = '2ea0736b2a4ccec4dfec32aabe070bbf4e15f40ecf6cfd83825053ae90906315'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(source_directory):
    notebook = json.loads((Path(source_directory)/TRAIN).read_text())
    tree = ast.parse(''.join(notebook['cells'][19]['source']))
    assert hashlib.sha256(ast.unparse(tree).encode()).hexdigest() == GENERATOR_PIN
    cls, = [n for n in tree.body if isinstance(n,ast.ClassDef)]
    method, = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '__getitem__']
    # Execute only the numerical batch method. Source exclusions and directory
    # enumeration never execute; all identities and planes below are synthetic.
    method_code = compile(ast.Module(body=[method],type_ignores=[]), '<source-batch-method>', 'exec')
    cases = []
    for sizes, dtype in [([1,3,7],np.float64), ([2],np.float32), ([4,1],np.float64)]:
        images = [np.stack([np.full((512,512,1), g*10+m+.25,dtype=dtype)
                           for m in range(size)]) for g,size in enumerate(sizes)]
        metadata = np.arange(len(sizes)*4,dtype=dtype).reshape(-1,4)/7
        slopes = np.linspace(-2.,3.,len(sizes),dtype=dtype)
        keys = ['synthetic_group_'+str(g) for g in range(len(sizes))]
        names = {k: ['synthetic_member_'+str(m) for m in range(sizes[g])]
                 for g,k in enumerate(keys)}
        planes = {(k,name):images[g][m,:,:,0] for g,k in enumerate(keys)
                  for m,name in enumerate(names[k])}
        reference_state = np.random.RandomState(735)
        candidate_state = np.random.RandomState(735)
        scope = dict(np=SimpleNamespace(array=np.array,expand_dims=np.expand_dims,
                     random=reference_state), get_img=lambda path:planes[tuple(path.split('/')[-2:])])
        exec(method_code, scope)
        population = SimpleNamespace(keys=keys,train_data=names,batch_size=8,
            a=dict(zip(keys,slopes)),tab=dict(zip(keys,metadata)))
        for batch in range(4):
            expected = scope['__getitem__'](population,batch)
            actual = sample_batch(images,metadata,slopes,candidate_state)
            for left,right in zip((*actual[0],actual[1]),(*expected[0],expected[1])):
                assert left.dtype == right.dtype
                np.testing.assert_array_equal(left,right)
        # Subsequent draws also agree: the whole RNG transition is preserved.
        np.testing.assert_array_equal(reference_state.randint(0,100000,size=20),
                                      candidate_state.randint(0,100000,size=20))
        cases.append(dict(groups=len(sizes),members=sum(sizes),batches=4,
                          dtype=np.dtype(dtype).name, exact_arrays_and_rng=True))
    reuse = []
    for domain, sizes, records in [
        ('synthetic_machine_measurements',[1,4,2],
         [np.arange(n*3).reshape(n,3)+g*100 for g,n in enumerate([1,4,2])]),
        ('synthetic_observatory_exposures',[3,1],
         [np.arange(n*2).reshape(n,2)/10+g for g,n in enumerate([3,1])])]:
        rng = np.random.RandomState(91)
        state = rng.get_state()
        groups,members = sample_indices(sizes,64,rng)
        values = np.stack([records[g][m] for g,m in zip(groups,members)])
        replay = np.random.RandomState()
        replay.set_state(state)
        groups2,members2 = sample_indices(sizes,64,replay)
        np.testing.assert_array_equal(groups,groups2)
        np.testing.assert_array_equal(members,members2)
        assert values.shape == (64,records[0].shape[1])
        assert np.all((groups>=0)&(groups<len(sizes)))
        assert all(0<=m<sizes[g] for g,m in zip(groups,members))
        reuse.append(dict(application=domain,draws=64,value_dimensions=values.shape[1],
                          replay_and_alignment_passed=True))
    rejected = 0
    for sizes,batch in [([],8),([0],8),([-1],8),([1.5],8),([True],8),([[2]],8),([2],0),([2],True)]:
        rng=np.random.RandomState(12)
        expected=np.random.RandomState(12)
        try:
            sample_indices(sizes,batch,rng)
        except ValueError:
            rejected+=1
        else:
            raise AssertionError('Invalid sampling contract accepted')
        np.testing.assert_array_equal(rng.randint(0,100000,size=20),expected.randint(0,100000,size=20))
    rejected_populations=0
    valid_images=[np.zeros((1,512,512,1),dtype=np.float32)]
    valid_metadata=np.zeros((1,4),dtype=np.float32)
    valid_slopes=np.zeros(1,dtype=np.float32)
    for images,metadata,slopes in [
        ([],valid_metadata,valid_slopes),
        ([np.zeros((1,4,4,1),dtype=np.float32)],valid_metadata,valid_slopes),
        (valid_images,np.full((1,4),np.nan,dtype=np.float32),valid_slopes),
        (valid_images,valid_metadata,np.zeros(2,dtype=np.float32))]:
        rng=np.random.RandomState(12)
        expected=np.random.RandomState(12)
        try:
            sample_batch(images,metadata,slopes,rng)
        except ValueError:
            rejected_populations+=1
        else:
            raise AssertionError('Invalid image population accepted')
        np.testing.assert_array_equal(rng.randint(0,100000,size=20),expected.randint(0,100000,size=20))
    return dict(format='osic-image-batches.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,source=dict(repository=REPOSITORY,revision=REVISION,
            normalized_generator_cell_sha256=GENERATOR_PIN),cases=cases,
        total_source_equivalent_samples=96,invalid_sampling_contracts_rejected=rejected,
        invalid_populations_rejected_without_rng_consumption=rejected_populations,
        constructed_reuse_examples=reuse,
        sampler_sha256=sha(ROOT/'sciona/group_member_sampling.py'),
        adapter_sha256=sha(ROOT/'sciona/osic_image_batches.py'),validator_sha256=sha(__file__),
        limitations=[
            'Valid, already eligible and preprocessed populations only; no decoding, exclusion-list or metadata preparation qualification.',
            'Corrupt source images can be silently skipped by the original method; this adapter instead rejects invalid supplied populations.',
            'Equal group probability is not equal probability across all members when group sizes differ.',
            'Explicit sequential RandomState calls do not reproduce historical Keras prefetch timing or its extra generator calls.',
            'Constructed reuse checks indexing and state transfer, not empirical domain performance or catalog approval.',
            'Full model training, source feature preparation and original workflow approval remain outstanding.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_image_batches.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_equivalent_samples=report['total_source_equivalent_samples'],
                         rejected=report['invalid_sampling_contracts_rejected'])))
