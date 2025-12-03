import numpy as np
import pandas as pd

def make_multitarget_strata(df: pd.DataFrame,
                            cols=None,
                            n_bins: int = 4,
                            n_splits: int = 5) -> np.ndarray:
    """
    Create a stratification label for multi-target regression.

    - Quantile-bin each target column into `n_bins`.
    - Combine the bins into a single integer code.
    - Reassign rare codes to the most common one so every class has >= n_splits samples.
    """
    if cols is None:
        cols = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]

    n = len(df)
    strat = np.zeros(n, dtype=int)

    for i, col in enumerate(cols):
        # qcut -> quantile bins 0..n_bins-1
        # duplicates="drop" handles constant columns
        bins = pd.qcut(df[col],
                       q=n_bins,
                       labels=False,
                       duplicates="drop")
        bins = bins.fillna(0).astype(int)  # in case of NaNs
        strat += bins.to_numpy() * (n_bins ** i)

    # Ensure each stratum has at least n_splits samples
    vc = pd.Series(strat).value_counts()

    # strata with < n_splits samples
    rare_labels = vc[vc < n_splits].index

    if len(rare_labels) > 0:
        # send all rare ones into the most frequent stratum
        majority_label = vc.idxmax()
        mask = np.isin(strat, rare_labels)
        strat[mask] = majority_label

    return strat
