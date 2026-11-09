"""Bounded immutable-stream lifecycle with validation checkpoint selection.

Private source descriptors never appear in the result. Training and validation
replay disk snapshots, not mutable source files. Validation selects an epoch;
its score must not be reported as independent generalization evidence.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from sciona import temporal_sparse_execution as base
from sciona.temporal_sparse_source import Source, events, snapshot, validate_time_splits
from sciona.temporal_sparse_selection import fit_selected
from sciona.validation_stopping import ValidationStopping


def selection_config(value):
    if type(value) is not dict or set(value) != {'max_epochs', 'patience', 'min_delta'}:
        raise ValueError('Explicit selection controls required')
    if type(value['max_epochs']) is not int or value['max_epochs'] < 1:
        raise ValueError('Positive integer epoch budget required')
    try:
        ValidationStopping(value['patience'], value['min_delta'])
        return json.dumps(value, allow_nan=False, sort_keys=True)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('Invalid selection controls') from None


@dataclass(frozen=True, repr=False)
class Prepared:
    source: base.Prepared = field(repr=False)
    selection: str = field(repr=False)


def prepare(payload):
    if type(payload) is not dict or 'selection' not in payload:
        raise ValueError('Selection recipe required')
    selection = selection_config(payload['selection'])
    configuration = {k: v for k, v in payload.items() if k != 'selection'}
    return Prepared(base.prepare(configuration), selection)


@contextmanager
def verified_streams(prepared):
    """Revalidate descriptors and all three snapshots before yielding inputs."""
    if type(prepared) is not base.Prepared:
        raise ValueError('Prepared source recipe required')
    try:
        config = json.loads(prepared.configuration)
        if base._configuration(config) != prepared.configuration:
            raise ValueError('Invalid prepared configuration')
    except (TypeError, ValueError):
        raise ValueError('Invalid prepared configuration') from None
    if type(prepared.sources) is not tuple or len(prepared.sources) != 3 or any(type(s) is not Source for s in prepared.sources):
        raise ValueError('Invalid prepared sources')
    limits = config['limits']
    for role, source in zip(('training', 'validation', 'query'), prepared.sources):
        if (source.path != str(Path(config['sources'][role]).absolute()) or
            type(source.labeled) is not bool or source.labeled != (role != 'query') or
            type(source.max_line_bytes) is not int or source.max_line_bytes != limits['max_line_bytes'] or
            type(source.max_bytes) is not int or source.max_bytes != limits['max_source_bytes']):
            raise ValueError('Prepared source configuration changed')
    with ExitStack() as stack:
        streams = [stack.enter_context(snapshot(s)) for s in prepared.sources]
        for source, stream in zip(prepared.sources, streams):
            count = 0
            first = last = None
            for event in events(stream, labeled=source.labeled, max_line_bytes=source.max_line_bytes):
                if first is None:
                    first = event['time']
                last = event['time']
                count += 1
            if not count or (count, first, last) != (source.rows, source.first_time, source.last_time):
                raise ValueError('Prepared source summary changed')
        validate_time_splits(*prepared.sources, gap=config['controls']['gap'])
        if prepared.sources[2].rows > limits['max_output_rows']:
            raise ValueError('Prediction output exceeds configured limit')

        def replay(index):
            streams[index].seek(0)
            source = prepared.sources[index]
            return events(streams[index], labeled=source.labeled, max_line_bytes=source.max_line_bytes)

        yield config, replay


def execute(prepared):
    if type(prepared) is not Prepared:
        raise ValueError('Prepared selection recipe required')
    try:
        selection = json.loads(prepared.selection)
        if selection_config(selection) != prepared.selection:
            raise ValueError('Invalid prepared selection')
    except (TypeError, ValueError):
        raise ValueError('Invalid prepared selection') from None
    with verified_streams(prepared.source) as (config, replay):
        selected = fit_selected(lambda: replay(0), lambda: replay(1),
            encoder_controls=config['encoder_controls'], controls=config['controls'], **selection)
        model = selected.model
        predictions = []
        for values, _ in model.predict_batches(replay(2)):
            if len(predictions) + len(values) > config['limits']['max_output_rows']:
                raise ValueError('Prediction output exceeds configured limit')
            predictions.extend(values.tolist())
        if not all(math.isfinite(v) for v in predictions):
            raise ValueError('Nonfinite predictions')
        return dict(training_rows=model.training_rows, query_rows=len(predictions),
            predictions=predictions, models=len(model.models), feature_count=8 + config['encoder_controls']['hash_size'],
            selected_epoch=selected.selected_epoch, evaluated_epochs=len(selected.validation_scores),
            stopped_early=selected.stopped_early, selected_updates_per_model=model.updates_per_model,
            selection_validation=dict(rows=prepared.source.sources[1].rows,
                rmsle=selected.validation_scores[selected.selected_epoch - 1],
                scores=list(selected.validation_scores), used_for_selection=True))
