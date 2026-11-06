"""Complete corrected uncertainty training, repeated inference and restoration."""
from datetime import date
import numpy as np
from sciona.m5u_preparation import prepare
from sciona.m5u_population import build
from sciona.m5u_training import train_batch,QUANTILES,PARAMETERS
from sciona.m5u_inference import forecast
from sciona.m5u_harmonization import restore


def run_unrestored(units,prices,roles,category_partitions,calendar,*,minimum_day=299,
        seed=514,speed=False,super_speed=False,capacity=3_000_000,progress=None):
    """Forecast 28 days after the last history day at every source level.

    The aggregate batch explicitly uses the retained-base-reference correction;
    base partitions preserve source normalization. Returned arrays are runtime
    data and must not be copied into public evidence. Reports contain counts.
    """
    if (type(seed) is not int or not 1<=seed<2**31-2000
            or type(minimum_day) is not int or minimum_day<0
            or type(speed) is not bool or type(super_speed) is not bool
            or not isinstance(category_partitions,dict) or set(category_partitions.values())!={13,14,15}):
        raise ValueError('Complete three-partition source configuration and valid controls required')
    units=np.asarray(units)
    if units.ndim!=2 or not all(units.shape):raise ValueError('Nonempty history required')
    dates=[date.fromisoformat(value) for value in calendar['dates']]
    if len(dates)<len(units)+28:raise ValueError('Complete 28-day future calendar required')
    predictions=[];indices=[];levels=[];factors=[];reports={}
    for controller in [-1,13,14,15]:
        prepared=prepare(units,prices,roles,category_partitions,controller_level=controller,
                         max_level=11 if controller==-1 else None,
                         normalization='retained_base_reference' if controller==-1 else 'source')
        population=build(prepared['history'],prepared['scaled_history'],prepared['revenue'],
                         prepared['roles'],prepared['levels'],[d.year for d in dates],
                         [d.month for d in dates],minimum_day)
        # Check the forecast origin before committing to native training work.
        if not (population['days']==len(units)-1).any():
            raise ValueError('Final history day is excluded by source calendar cleaning')
        fitted,training=train_batch(prepared,population,calendar,controller,
                                    speed=speed,super_speed=super_speed,seed=seed+controller)
        if progress:progress(dict(stage='trained',controller=controller,models=sum(r['models'] for r in training.values())))
        for level,models in fitted.items():
            forecasted=forecast(models,population,prepared['history'],calendar,level,len(units)-1,QUANTILES,
                                scale_range=training[level]['scale_range'],speed=speed,super_speed=super_speed,
                                seed=seed+1000+level,capacity=capacity)
            selected=forecasted['series']
            predictions.append(forecasted['predictions'])
            indices.append(prepared['hierarchy_indices'][selected])
            levels.append(prepared['levels'][selected]);factors.append(prepared['factors'][selected])
            reports[level]=dict(training[level],query_rows=forecasted['query_rows'],query_batches=forecasted['batches'])
            if progress:progress(dict(stage='forecast',level=level,query_rows=forecasted['query_rows']))
    if set(reports)!=set(PARAMETERS):raise ValueError('Incomplete source level inventory')
    indices=np.concatenate(indices);order=np.argsort(indices)
    if not np.array_equal(indices[order],np.arange(len(indices))):raise ValueError('Hierarchy coverage is incomplete or duplicated')
    levels=np.concatenate(levels)[order];factors=np.concatenate(factors)[order]
    unadjusted=np.concatenate(predictions,axis=2)[:,:,order]
    return dict(unadjusted=unadjusted,factors=factors,adjustment=1. if speed or super_speed else .7,
                levels=levels,quantiles=QUANTILES,
                hierarchy_indices=indices[order],reports=reports,
                normalization='retained_base_reference',scope='Corrected independent reconstruction; no historical-score parity claim.')


def run(units,prices,roles,category_partitions,calendar,*,minimum_day=299,
        seed=514,speed=False,super_speed=False,capacity=3_000_000,progress=None):
    """Retain the complete reference lifecycle and original restoration arithmetic."""
    result=run_unrestored(units,prices,roles,category_partitions,calendar,
        minimum_day=minimum_day,seed=seed,speed=speed,super_speed=super_speed,
        capacity=capacity,progress=progress)
    restored=restore(result['unadjusted'],result['levels'],result['factors'],result['quantiles'],
                     adjustment=result['adjustment'])
    return dict(predictions=restored,**{key:result[key] for key in
        ('levels','quantiles','hierarchy_indices','reports','normalization','scope')})
