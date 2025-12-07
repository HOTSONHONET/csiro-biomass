import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

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


def make_state_date_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    state_col: str = "State",
    date_col: str = "Sampling_Date",
) -> pd.DataFrame:
    """
    Create folds using GroupKFold with groups = State + Sampling_Date.

    Returns a copy of df with a new column:
      - 'fold' : integer fold id in [0, n_splits-1]
    """
    df = df.copy()

    # group label: "State_SamplingDate"
    df["env_group"] = df[state_col].astype(str) + "_" + df[date_col].astype(str)

    gkf = GroupKFold(n_splits=n_splits)

    df["fold"] = -1
    groups = df["env_group"].values

    for fold_id, (_, val_idx) in enumerate(gkf.split(df, groups=groups)):
        df.loc[df.index[val_idx], "fold"] = fold_id

    return df

def make_state_date_stratified_group_folds(
    df: pd.DataFrame,
    target_cols=None,
    n_bins: int = 4,
    n_splits: int = 5,
    seed: int = 42,
    state_col: str = "State",
    date_col: str = "Sampling_Date",
) -> pd.DataFrame:
    """
    Create folds using StratifiedGroupKFold with:
      - groups  = State + Sampling_Date
      - strata  = multi-target bins over target_cols (via make_multitarget_strata)

    Returns a copy of df with a new column:
      - 'fold' : integer fold id in [0, n_splits-1]
    """

    df = df.copy()

    # if no targets specified, use the 5 biomass ones by default
    if target_cols is None:
        target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

    # ensure numeric
    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) strata from multi-target bins
    df["strata"] = make_multitarget_strata(
        df,
        cols=target_cols,
        n_bins=n_bins,
        n_splits=n_splits,
    )

    # 2) group label: "State_SamplingDate"
    df["env_group"] = df[state_col].astype(str) + "_" + df[date_col].astype(str)

    # 3) Stratified + Grouped split
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    df["fold"] = -1
    X_dummy = np.zeros(len(df), dtype=np.int8)

    for fold_id, (_, val_idx) in enumerate(
        sgkf.split(X_dummy, df["strata"].values, groups=df["env_group"].values)
    ):
        df.loc[df.index[val_idx], "fold"] = fold_id

    return df