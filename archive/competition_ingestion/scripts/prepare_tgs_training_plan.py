"""Read-only expansion of reviewed TGS source phase inventory into fit dependencies."""
import hashlib
import json
from pathlib import Path
from sciona.tgs_checkpoint_selection import ensemble_selection


def build_plan(root):
    inventory_path = root / 'docs/reviews/competition_tgs_training_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    fits = []
    keras_first = [p for p in inventory['phases'] if p['source_script'] == 'bes/stage1.sh']
    keras_second = [p for p in inventory['phases'] if p['source_script'] == 'bes/stage2_train.sh']
    assert len(keras_first) == 5 and len(keras_second) == 6
    for group, phases in ((1, keras_first), (2, keras_second)):
        for phase, entry in enumerate(phases):
            for fold in map(int, entry['controls']['fold'].split(',')):
                key = f'keras.r{group}.p{phase}.f{fold}'
                predecessor = None if phase == 0 else f'keras.r{group}.p{phase-1}.f{0 if group==2 and phase<=2 else fold}'
                fits.append(dict(key=key, branch='keras', controls=entry['controls'], fold=fold,
                                 weight_predecessor=predecessor, initial_reference='resnext50' if predecessor is None else None,
                                 weight_selection='best_validation', pseudo_round=1 if group==2 else None,
                                 population='pseudo_only' if group==2 and phase<2 else 'labeled_fold',
                                 training_selection='confident_pseudo_shuffle13_nonconstant' if group==2 and phase<2 else 'labeled_fold_complement_nonconstant',
                                 validation_selection='labeled_same_fold_nonconstant',
                                 epoch_ceiling=entry['epochs_per_fit']))
    torch_phases = [p for p in inventory['phases'] if p['source_script'].startswith('phalanx/')]
    assert len(torch_phases) == 4
    for phase, entry in enumerate(torch_phases):
        for fold in range(entry['fold_fits']):
            controls = dict(entry['controls'])
            controls['checkpoint_policy'] = 'cycle_end' if phase == 2 else 'cycle_best'
            fits.append(dict(key=f'torch.p{phase}.f{fold}', branch='pytorch', controls=controls, fold=fold,
                             weight_predecessor='torch.p2.f0' if phase==3 else None,
                             initial_reference='resnet34' if phase!=3 else None,
                             weight_selection='final_cycle_end' if phase==2 else 'cycle_best',
                             pseudo_round=None if phase==0 else (1 if phase==1 else 2),
                             population='labeled_plus_pseudo_fold' if phase==1 else ('pseudo_only' if phase==2 else 'labeled_fold'),
                             training_selection={0:'labeled_fold_complement', 1:'labeled_fold_complement_plus_same_index_pseudo_bucket',
                                                 2:'entire_query_population', 3:'labeled_fold_complement'}[phase],
                             validation_selection='explicit_labeled_validation_indices' if phase==2 else 'labeled_same_fold',
                             epoch_ceiling=entry['epochs_per_fit']))
    assert len(fits) == inventory['total_phase_fits'] == 63
    keys = {fit['key'] for fit in fits}
    assert len(keys) == len(fits)
    assert all(f['weight_predecessor'] is None or f['weight_predecessor'] in keys for f in fits)
    assert sum(f['epoch_ceiling'] for f in fits) == inventory['total_fold_epochs'] == 6530
    ensembles = [ensemble_selection(stage) for stage in (1, 2, 3)]
    for ensemble in ensembles:
        required = [item['fit'] for snapshot in ensemble['keras'] for item in snapshot]
        required += [item['fit'] for item in ensemble['pytorch']]
        assert set(required) <= keys
        ensemble['requires_fits'] = required
    return dict(approved=False, catalog_mutations=0, full_training_executed=False, fits=fits,
                ensembles=ensembles,
                execution_barriers=[
                    dict(key='pseudo.r1', requires_ensemble=1, releases=['keras.r2', 'torch.p1']),
                    dict(key='pseudo.r2', requires_ensemble=2, releases=['torch.p2']),
                    dict(key='final', requires_ensemble=3, requires_mosaic=True)],
                fit_count=len(fits), epoch_ceiling=sum(f['epoch_ceiling'] for f in fits),
                initialization_policy='Each fold receives independent source initialization or exact predecessor copy; no cross-fold model carryover.',
                initialization_evidence='docs/reviews/competition_tgs_initialization.json',
                unresolved=['Qualify pretrained reference reuse and historical runtime differences.',
                            'Supply explicit branch fold assignments and PyTorch pseudo-only validation indices; never infer the unpublished split.',
                            'Execute complete source budgets and serialize verified graph.'],
                inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                selection_sha256=hashlib.sha256((root/'sciona/tgs_checkpoint_selection.py').read_bytes()).hexdigest(),
                planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = build_plan(root)
    (root/'docs/reviews/competition_tgs_training_plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='fits'},indent=2))
