"""Synthetic-only shard boundaries, complete iteration and isolation."""
import json
import pytest
from sciona.video_multilabel_shards import FeatureShards, validate_populations


def record(i, labelled=True):
    value = dict(frames=[[float(i), 1.] for _ in range(i % 3 + 1)], sparse={'feature':float(i)}, group='synthetic-'+str(i))
    if labelled:
        value['labels'] = [i % 2, (i+1) % 2]
    return value


def write(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_complete_ordered_and_shuffled_epochs(tmp_path):
    rows = [record(i) for i in range(7)]
    paths = [write(tmp_path/'a.jsonl', rows[:3]), write(tmp_path/'b.jsonl', rows[3:])]
    store = FeatureShards(paths, labelled=True)
    assert list(store) == rows
    for order in ([6,0,5,1,4,2,3], list(range(7))):
        batches = list(store.batches(3, order))
        assert [len(b) for b in batches] == [3,3,1]
        assert [r for b in batches for r in b] == [rows[i] for i in order]
    assert store.labels == tuple(tuple(r['labels']) for r in rows)
    store[0]['frames'][0][0] = 999
    assert store[0] == rows[0]
    with pytest.raises(TypeError):
        store[:]


def test_lazy_batch_reads(tmp_path, monkeypatch):
    store = FeatureShards([write(tmp_path/'a.jsonl', [record(i) for i in range(7)])], labelled=True)
    reads = []
    original = FeatureShards.__getitem__
    def observed(self, index):
        reads.append(index)
        return original(self, index)
    monkeypatch.setattr(FeatureShards, '__getitem__', observed)
    iterator = store.batches(2)
    assert reads == []
    assert len(next(iterator)) == 2 and reads == [0,1]
    assert not any(isinstance(v,dict) for v in store.__dict__.values())


def test_split_isolation_and_query_labels(tmp_path):
    train = FeatureShards([write(tmp_path/'a.jsonl', [record(0),record(1)])], labelled=True)
    calibration = FeatureShards([write(tmp_path/'b.jsonl', [record(2),record(3)])], labelled=True)
    query = FeatureShards([write(tmp_path/'c.jsonl', [record(4,False)])], labelled=False)
    validate_populations(train, calibration, query)
    overlap = FeatureShards([write(tmp_path/'d.jsonl', [record(1,False)])], labelled=False)
    with pytest.raises(ValueError, match='overlap'):validate_populations(train, calibration, overlap)
    with pytest.raises(ValueError, match='fields'):FeatureShards([write(tmp_path/'e.jsonl', [record(5)])], labelled=False)


def test_mutation_and_duplicate_shards(tmp_path):
    path = write(tmp_path/'a.jsonl', [record(0)])
    store = FeatureShards([path], labelled=True)
    with pytest.raises(ValueError, match='Repeated'):FeatureShards([path,path], labelled=True)
    write(path, [record(1)])
    with pytest.raises(ValueError, match='changed'):store[0]


@pytest.mark.parametrize('raw', ['{}\n', '{"frames":[],"frames":[]}\n', 'invalid\n', '\n'])
def test_invalid_records(tmp_path, raw):
    path=tmp_path/'a.jsonl';path.write_text(raw)
    with pytest.raises(ValueError):FeatureShards([path], labelled=True)


def test_record_limit_and_empty_population(tmp_path):
    path=write(tmp_path/'a.jsonl',[record(0)])
    with pytest.raises(ValueError, match='byte limit'):FeatureShards([path], labelled=True, max_record_bytes=8)
    path.write_text('')
    with pytest.raises(ValueError, match='Empty'):FeatureShards([path], labelled=True)


def test_oversized_line_read_is_bounded(tmp_path, monkeypatch):
    from pathlib import Path
    path=tmp_path/'a.jsonl';path.write_bytes(b'x'*10000)
    original=Path.open
    requests=[]
    class Observed:
        def __init__(self, stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def fileno(self):return self.stream.fileno()
        def tell(self):return self.stream.tell()
        def readline(self, size):
            requests.append(size)
            return self.stream.readline(size)
    monkeypatch.setattr(Path,'open',lambda p,*a,**k:Observed(original(p,*a,**k)))
    with pytest.raises(ValueError, match='byte limit'):
        FeatureShards([path],labelled=True,max_record_bytes=32)
    assert requests==[33]


def test_width_mismatch_across_shards(tmp_path):
    changed=record(1);changed['frames']=[[1.,2.,3.]]
    with pytest.raises(ValueError, match='widths'):
        FeatureShards([write(tmp_path/'a.jsonl',[record(0)]),write(tmp_path/'b.jsonl',[changed])],labelled=True)


@pytest.mark.parametrize('field,value', [('frames',[[float('nan'),1.]]),('frames',[[1.,2.],[3.]]),('labels',[True,0]),('sparse',{'feature':-1.})])
def test_invalid_values(tmp_path, field, value):
    row=record(0);row[field]=value
    with pytest.raises(ValueError):FeatureShards([write(tmp_path/'a.jsonl',[row])], labelled=True)
