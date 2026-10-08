"""A response that produced nothing scoreable counts as wrong (0), not missing.

This applies to a response that failed to parse and to one that parsed but
omitted or left empty the field a metric needs. Every dimension therefore has
the same denominator, all 2,917 puzzles. Averaging over successful responses
instead would make a score depend on each model's parse rate (73% to 100%
across the 14 models).

It does not apply where the ground truth is missing: callers pass `applicable`
to exclude those rows, which stay NaN and drop out of the mean (e.g. concept
coverage when the reference decomposes to zero claims).
"""

import numpy as np
import pandas as pd

FAILURE_SCORE = 0.0


def count_failures_as_wrong(df: pd.DataFrame, columns, applicable=None) -> pd.DataFrame:
    """Fill NaN in `columns` with FAILURE_SCORE, in place; return df.

    `applicable`: optional boolean mask aligned with df -- rows where the
    metric can be scored at all. Rows outside it keep NaN.
    """
    for col in columns:
        missing = df[col].isna()
        if applicable is not None:
            missing &= np.asarray(applicable, dtype=bool)
        df.loc[missing, col] = FAILURE_SCORE
    return df
