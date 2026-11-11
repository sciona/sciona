"""Synthetic replay of pinned source save decisions and inherited fallback."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace

from sciona.wheat_checkpoint_selection import PseudoCheckpointSelection


def main(runtime_source, notebook_source, output):
    raw = runtime_source.read_bytes()
    notebook = notebook_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '36551e0efd6b7d60778d2c4965a7dc4936e7101937bcde1cda4aeef1b9f5418b':
        raise ValueError('runtime reference differs')
    if hashlib.sha256(notebook).hexdigest() != '3ffd2f6dbc30822d6aeb104dd1ea6bae5856ed3f2c01a639f7fee27867d111ab':
        raise ValueError('notebook reference differs')
    tree = ast.parse(raw)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'val_loss < val_loss_min')
    code = compile(ast.Module(body=[node], type_ignores=[]), '<pinned-wheat-save-decision>', 'exec')
    text = notebook.decode()
    conversion = re.search(r'^\s*refine_checkpoint_out\(([^\n]+)\)', text, re.M)
    if conversion is None:
        raise ValueError('round-one inference checkpoint conversion absent')
    call = ast.parse('refine_checkpoint_out(' + conversion.group(1) + ')', mode='eval').body
    inherited_path = ast.literal_eval(call.args[1])
    destinations = re.findall(r'--checkpoint-path\s+(\S+)', text)
    if len(destinations) != 2 or destinations[1] != inherited_path:
        raise ValueError('notebook does not retain round-one inference checkpoint as round-two destination')
    rng = random.Random(83)
    cases = 0
    inherited_cases = 0
    for round_number in (1, 2):
        for scenario in range(64):
            previous_loss = None if round_number == 1 else 2.
            selection = PseudoCheckpointSelection(round_number=round_number, inherited_best_loss=previous_loss)
            saved = []
            namespace = dict(val_loss_min=float('inf') if previous_loss is None else previous_loss,
                args=SimpleNamespace(save_optimizer=round_number == 1), ckpt_path='synthetic-checkpoint',
                model=SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {'synthetic_epoch': namespace['epoch']})),
                optimizer=SimpleNamespace(state_dict=lambda: {'synthetic_optimizer': True}),
                torch=SimpleNamespace(save=lambda payload, path: saved.append(payload)), print=lambda *args: None)
            expected_epoch = None
            for epoch in range(selection.epochs):
                loss = (2. if scenario == 0 else 3. if scenario == 1 else rng.choice([.5, 1., 2., 3., 4.]))
                namespace.update(val_loss=loss, epoch=epoch)
                before = len(saved)
                exec(code, namespace)
                source_save = len(saved) > before
                assert selection.observe(epoch, loss) == source_save
                if source_save:
                    expected_epoch = epoch
                    payload = saved[-1]
                    state = payload['model'] if round_number == 1 else payload
                    assert state['synthetic_epoch'] == epoch
                    if round_number == 1:
                        assert payload['val_loss_min'] == loss and 'optimizer' in payload
            result = selection.result()
            assert result['selected_epoch'] == expected_epoch
            assert result['selected_validation_loss'] == namespace['val_loss_min']
            if expected_epoch is None:
                assert round_number == 2 and result['selected_origin'] == 'inherited_round1'
                inherited_cases += 1
            cases += 1
    assert inherited_cases >= 2
    files = ['sciona/wheat_checkpoint_selection.py', 'scripts/validate_wheat_checkpoint_selection.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_decision_sequences_checked=cases, inherited_fallback_cases=inherited_cases,
        notebook_round2_destination_matches_round1_inference_artifact=True,
        source_runtime_sha256=hashlib.sha256(raw).hexdigest(),
        source_notebook_sha256=hashlib.sha256(notebook).hexdigest(),
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Selection and save-payload structure only; actual tensor/optimizer checkpoint provenance remains a caller responsibility.',
                'Nonfinite validation metrics are rejected by reconstruction even though source comparison could silently ignore them.',
                'Full training, inference, artifact qualification and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'implementation_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-source', type=Path, required=True)
    parser.add_argument('--notebook-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.runtime_source, args.notebook_source, args.output)
