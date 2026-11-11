"""Source fold exclusion and two-stage auxiliary populations for base fits."""
import numpy as np
import pandas as pd


def base_populations(primary, auxiliary, warm_only, fold, *, numpy_rng):
    """Return shuffled warm/training rows and positive-only validation rows.

    All tables are caller-supplied runtime inputs. The first 20 epochs use
    both auxiliary populations; subsequent epochs omit the warm-only group.
    Shuffling consumes one shared legacy NumPy RNG in source order.
    """
    if type(fold) is not int or fold not in range(5) or not isinstance(numpy_rng, np.random.RandomState):
        raise ValueError('source fold and explicit legacy NumPy RNG required')
    if any(not isinstance(table, pd.DataFrame) for table in (primary, auxiliary, warm_only)):
        raise ValueError('three runtime annotation tables required')
    if not {'fold', 'isbox'}.issubset(primary.columns):
        raise ValueError('primary fold and box eligibility required')
    validation = primary.loc[primary['fold'] == fold]
    training = primary.loc[~primary.index.isin(validation.index)]
    validation = validation.loc[validation['isbox'] == True].reset_index(drop=True)
    warm = pd.concat([training, auxiliary, warm_only], ignore_index=True).sample(frac=1, random_state=numpy_rng).reset_index(drop=True)
    training = pd.concat([training, auxiliary], ignore_index=True).sample(frac=1, random_state=numpy_rng).reset_index(drop=True)
    return dict(warm=warm, training=training, validation=validation)
