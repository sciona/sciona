"""Dependency scheduling for the full reviewed TGS fit and ensemble inventory.

Scheduling is not evidence of training. Callers mark actions complete only
after verifying and durably recording their actual execution results.
"""


def next_action(plan, completed_fits, completed_rounds):
    fits = plan['fits']
    by_key = {fit['key']: fit for fit in fits}
    if len(fits) != 63 or len(by_key) != 63 or sum(f['epoch_ceiling'] for f in fits) != 6530:
        raise ValueError('complete reviewed 63-fit inventory required')
    done, rounds = set(completed_fits), set(completed_rounds)
    if not done <= set(by_key) or any(type(stage) is not int or stage not in (1, 2, 3) for stage in rounds):
        raise ValueError('unknown completed action')
    ensembles = {entry['stage']: entry for entry in plan['ensembles']}
    if set(ensembles) != {1, 2, 3}:
        raise ValueError('all three ensemble rounds required')
    for key in done:
        fit = by_key[key]
        if (fit['weight_predecessor'] is not None and fit['weight_predecessor'] not in done
                or fit['pseudo_round'] is not None and fit['pseudo_round'] not in rounds):
            raise ValueError('completed fit lacks completed dependencies: ' + key)
    for stage in rounds:
        if not set(ensembles[stage]['requires_fits']) <= done or (stage > 1 and stage - 1 not in rounds):
            raise ValueError('completed ensemble lacks completed dependencies')
    for stage in (1, 2, 3):
        if (stage not in rounds and (stage == 1 or stage - 1 in rounds)
                and set(ensembles[stage]['requires_fits']) <= done):
            return dict(kind='ensemble', stage=stage)
    for fit in fits:
        if (fit['key'] not in done
                and (fit['weight_predecessor'] is None or fit['weight_predecessor'] in done)
                and (fit['pseudo_round'] is None or fit['pseudo_round'] in rounds)):
            return dict(kind='fit', key=fit['key'])
    if len(done) == 63 and rounds == {1, 2, 3}:
        return dict(kind='complete')
    raise ValueError('incomplete workflow has no runnable action')
