import numpy as np
from .mice import MICE


class _Completion:
    def __init__(self, x, missing=(np.Inf, np.nan)):
        self.x = np.array(x)
        self.missing = list(missing)

    def _mice_completion(self, impute_type, n_nearest_columns=np.Inf):
        X_filled = MICE(impute_type=impute_type,
                        n_nearest_columns=n_nearest_columns).complete(self.x)
        return X_filled


def completion(x, mode, impute_type="col", missing=(np.nan, np.Inf)):
    if mode == "mice":
        return _Completion(x, missing)._mice_completion(impute_type)
