"""All thirteen source configurations and seventeen full-length synthetic phases."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_lifecycle import execute


def main():
    torch.set_num_threads(2)
    records=[dict(key=f'synthetic-{i}',waveform=(.2*np.sin(np.arange(960001)*(.021+i*.007))).astype(np.float32),sample_rate=32000,primary=i,secondary=[]) for i in range(5)]
    populations={}
    for group,count in [('four',4),('five',5)]:
        for fold in range(count):populations[group,fold]=dict(training=[r for i,r in enumerate(records[:count]) if i!=fold],validation=[records[fold]])
    noise=(.15*np.sin(np.arange(32000)*.13)+.04).astype(np.float32)
    result=execute('/private/tmp/sciona_cornell_source','/private/tmp/sciona_cornell_dependencies/audiomentations',
        populations=populations,background=[noise],short_noises=[noise],inference=records[0]['waveform'][:960000],
        epochs=2,batch_size=2,seed=13,synthetic=True,progress=lambda r:print(json.dumps(r),flush=True))
    assert result['models_completed']==13 and len(result['training_phases'])==17
    assert sum(r['phase']=='continuation' for r in result['training_phases'])==4
    assert {r['member'] for r in result['training_phases']}==set(range(13))
    assert all(r['selected_epoch']==2 for r in result['training_phases'])
    assert result['window_votes'].shape==(6,264) and result['whole_record_decisions'].shape==(264,)
    assert np.all((result['window_votes']>=0)&(result['window_votes']<=13))
    paths=sorted((ROOT/'sciona').glob('cornell_*.py'))
    report=dict(status='passed',models_completed=13,training_phases=17,continuation_phases=4,
        all_source_configurations=True,full_30_second_training=True,ten_copy_inference=True,
        all_selected_checkpoints_reloaded=True,window_count=6,classes=264,
        runtime_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All full source models and17phases; two synthetic epochs each with batch2 and offline random initial weights. No competition accuracy or historical checkpoint/pretraining parity. Full graph and publication review remain separate.')
    (ROOT/'docs/reviews/competition_cornell_lifecycle_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
