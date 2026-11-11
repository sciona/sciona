"""Read-only completion prerequisites for a synthetic run affected by relocation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REVISION='5ed15bda'
FILES=('sciona/group_member_sampling.py','sciona/keras_serial_steps.py',
       'sciona/osic_image_batches.py','sciona/osic_image_model.py',
       'sciona/osic_image_training_controls.py','sciona/osic_image_training.py',
       'scripts/run_osic_synthetic_image_training.py')


def check_progress(rows):
    if len(rows)>30:raise ValueError('Unexpected additional epochs')
    for epoch,row in enumerate(rows,1):
        if row['epoch']!=epoch or row['optimizer_iterations']!=epoch*32:
            raise ValueError('Noncontiguous epochs or incorrect update count')
        for key in ('loss','val_loss','learning_rate'):
            if type(row[key]) not in (int,float) or not math.isfinite(row[key]):
                raise ValueError('Nonfinite progress')
        if row['learning_rate']<=0:raise ValueError('Nonpositive learning rate')
    return len(rows)==30


def audit(directory):
    launch=json.loads((directory/'launch.json').read_text())
    expected={name:hashlib.sha256(subprocess.check_output(['git','show',REVISION+':'+name],cwd=ROOT)).hexdigest()
              for name in FILES}
    if launch['code_sha256']!=expected or launch.get('synthetic_only') is not True:
        raise ValueError('Frozen synthetic launch identities required')
    for key,value in dict(epochs=30,batch_size=8,training_steps=32,validation_steps=16,
                          training_seed=517,fixture_seed=805,native_threads=1).items():
        if launch[key]!=value:raise ValueError('Synthetic launch contract differs')
    raw=(directory/'progress.jsonl').read_bytes()
    # A live writer may have an unfinished trailing line. Never count it as a
    # completed epoch, and do not infer a terminal process state from the file.
    complete_lines=raw.split(b'\n')[:-1]
    rows=[json.loads(line) for line in complete_lines]
    complete=check_progress(rows)
    return dict(format='osic-training-finalization-audit.v1',read_only=True,approved=False,
        catalog_mutations=0,synthetic_only=True,frozen_launch_verified=True,source_revision=REVISION,
        completed_epochs=len(rows),observed_optimizer_updates=rows[-1]['optimizer_iterations'] if rows else 0,
        complete_schedule_observed=complete,partial_progress_line=bool(raw and not raw.endswith(b'\n')),
        checkpoint_present=(directory/'checkpoints/best.h5').is_file(),
        replay_inputs_present=(directory/'replay.npz').is_file(),
        replay_report_present=(directory/'replay.json').is_file(),
        original_result_present=(directory/'result.json').is_file(),
        limitations=['Progress and file-presence audit only; no checkpoint loading, replay, provenance authentication or training approval.',
            'File absence does not establish worker termination. Poll the live process before any finalization action.',
            'A migrated source tree cannot satisfy the original end-of-run source-path identity check.',
            'Separate frozen-code replay and explicit relocation accounting are required before assessing complete training evidence.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.runtime_directory),indent=2))
