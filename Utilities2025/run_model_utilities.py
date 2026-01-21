import pandas as pd 
import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Tuple
from pybats.dglm import bin_dglm
from typing import Dict, Any
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report
)
import copy

def make_day_slot_matrices(y: pd.Series, n: pd.Series, freq: str = "5min"):
    """
    Reshape y (counts) and n (sizes) into (day x slots) matrices.
    freq: string like '5min' or '15min' (must divide 1 day evenly).
    """
    y = y.sort_index()
    n = n.reindex_like(y).fillna(0)
    
    idx = pd.to_datetime(y.index)
    days = idx.normalize()
    
    # slots per day from freq
    slots_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    
    # slot number for each timestamp
    def slot_of_day(ts: pd.Timestamp) -> int:
        return int((ts - ts.normalize()) / pd.Timedelta(freq))
    
    slots = np.fromiter((slot_of_day(t) for t in idx), count=len(idx), dtype=int)
    
    days_index = pd.DatetimeIndex(sorted(days.unique()))
    D, S = len(days_index), slots_per_day
    
    # map day -> row
    day_to_row = {d: i for i, d in enumerate(days_index)}
    rows = np.array([day_to_row[d] for d in days], dtype=int)
    cols = slots
    
    Y = np.full((D, S), np.nan, dtype=float)
    N = np.full((D, S), np.nan, dtype=float)
    Y[rows, cols] = y.to_numpy(dtype=float)
    N[rows, cols] = n.to_numpy(dtype=float)
    
    # mask for valid targets (N>0)
    target_mask = np.isfinite(N) & (N > 0)
    
    # matrix of timestamps
    times_matrix = np.empty((D, S), dtype='datetime64[ns]')
    for i, d in enumerate(days_index):
        times_matrix[i, :] = pd.date_range(d, periods=S, freq=freq).to_numpy()
    
    return Y, N, days_index, times_matrix, target_mask, slots_per_day

    idx = pd.to_datetime(y.index)
    days = idx.normalize()

    # slots per day from freq
    slots_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))

    # slot number for each timestamp
    def slot_of_day(ts: pd.Timestamp) -> int:
        return int((ts - ts.normalize()) / pd.Timedelta(freq))

    slots = np.fromiter((slot_of_day(t) for t in idx), count=len(idx), dtype=int)

    days_index = pd.DatetimeIndex(sorted(days.unique()))
    D, S = len(days_index), slots_per_day

    # map day -> row
    day_to_row = {d: i for i, d in enumerate(days_index)}
    rows = np.array([day_to_row[d] for d in days], dtype=int)
    cols = slots

    Y = np.full((D, S), np.nan, dtype=float)
    N = np.full((D, S), np.nan, dtype=float)
    Y[rows, cols] = y.to_numpy(dtype=float)
    N[rows, cols] = n.to_numpy(dtype=float)

    # mask for valid targets (N>0)
    target_mask = np.isfinite(N) & (N > 0)

    # matrix of timestamps
    times_matrix = np.empty((D, S), dtype='datetime64[ns]')
    for i, d in enumerate(days_index):
        times_matrix[i, :] = pd.date_range(d, periods=S, freq=freq).to_numpy()

    return Y, N, days_index, times_matrix, target_mask, slots_per_day


def kernel_forecast_cumulative_matrix(
    y: pd.Series, n: pd.Series,
    freq: str = "5min",
    h_minutes: int = 30,
    window_minutes: int = 60,
    lambda_days: float = 5.0,
    include_same_day: bool = True,
    alpha0: float = 1.0, beta0: float = 1.0
) -> pd.Series:
    """
    Kernel-smoothed cumulative forecast for binomial probs.
    - y, n: pd.Series with DatetimeIndex
    - freq: time resolution string ('5min' default)
    """
    Y, N, days_index, times_mat, target_mask, S = make_day_slot_matrices(y, n, freq=freq)
    D = Y.shape[0]
    
    valid = np.isfinite(N) & (N > 0)
    
    # convert params to slots
    slot_minutes = pd.Timedelta(freq).seconds / 60
    h_bins = max(1, int(round(h_minutes / slot_minutes)))
    W_bins = max(1, int(round(window_minutes / slot_minutes)))
    
    P = np.full((D, S), np.nan, dtype=float)
    for d in range(D):
        # row (day) weights
        day_lag = (d - np.arange(D))
        w_row = np.exp(-day_lag / lambda_days)
        w_row[(day_lag < 0)] = 0.0
        if not include_same_day:
            w_row[day_lag == 0] = 0.0
        w_row = w_row.reshape(-1, 1)
        
        for s in range(S):
            if not target_mask[d, s]:
                continue
            
            delta = np.abs(np.arange(S) - s)
            delta = np.minimum(delta, S - delta)
            w_col = np.exp(-(delta / h_bins) ** 2)
            w_col[delta > W_bins] = 0.0
            w_col = w_col.reshape(1, -1)
            
            W = w_row * w_col * valid
            if include_same_day:
                W[d, s:] = 0.0
            
            if not np.any(W):
                P[d, s] = alpha0 / (alpha0 + beta0)
                continue
            
            Ynz = np.nan_to_num(Y)
            Nnz = np.nan_to_num(N)
            alpha = alpha0 + np.sum(W * Ynz)
            beta = beta0 + np.sum(W * (Nnz - Ynz))
            P[d, s] = alpha / (alpha + beta)
    
    # Map back to original timestamps
    ts_to_val = {pd.Timestamp(times_mat[i, j]): P[i, j]
                 for i in range(D) for j in range(S)}
    idx = pd.to_datetime(y.index)
    return pd.Series([ts_to_val.get(pd.Timestamp(t), np.nan) for t in idx], index=idx, name="p_hat_kernel_matrix")

def kernel_smoothed_transition_probs(
    transition_counts: pd.DataFrame,
    size_counts: pd.DataFrame,
    freq: str = "5min",
    h_minutes: int = 5,
    window_minutes: int = 15,
    include_same_day: bool = True
) -> pd.DataFrame:
    """
    Apply kernel_forecast_cumulative_matrix row-wise to get smoothed transition probabilities.
    
    Parameters
    ----------
    transition_counts : DataFrame
        Rows = (O,D) MultiIndex, Cols = time (DatetimeIndex)
    size_counts : DataFrame
        Same shape as transition_counts, binomial sizes n_t
    freq : str
        Time resolution of the series (default "5min").
    h_minutes : int
        Kernel bandwidth parameter.
    window_minutes : int
        Half-window size around target time to aggregate.
    verbose : bool
        Print progress every 100 rows.
    
    Returns
    -------
    DataFrame
        Smoothed probabilities with same index/columns as transition_counts.
    """
    nrows, ncols = transition_counts.shape
    out = np.zeros((nrows, ncols), dtype=float)
    
    for i in range(nrows):
        y_series = transition_counts.iloc[i, :]
        n_series = size_counts.iloc[i, :]
        
        smoothed = kernel_forecast_cumulative_matrix(
            y_series, n_series,
            freq=freq,
            h_minutes=h_minutes,
            window_minutes=window_minutes,
            include_same_day = include_same_day
        )
        out[i, :] = smoothed.values
    
    return pd.DataFrame(out, index=transition_counts.index, columns=transition_counts.columns)
    
def make_simple_time_indicators(
    tindex: pd.DatetimeIndex,
    day_start: str = "07:00",   # start of "day"
    day_end: str   = "19:00",   # end of "day"
    prefix: str = "ind_"
) -> pd.DataFrame:
    """
    Create simple time indicators for a DatetimeIndex:
      - is_day / is_night (based on day_start/day_end window)
      - is_weekday / is_weekend (weekday=Mon-Fri, weekend=Sat/Sun)
    """
    if not isinstance(tindex, pd.DatetimeIndex):
        tindex = pd.to_datetime(tindex)

    df = pd.DataFrame(index=tindex)

    # Day/night window
    start = pd.to_datetime(day_start).time()
    end   = pd.to_datetime(day_end).time()
    tt = tindex.time
    if start < end:
        is_day = (tt >= start) & (tt < end)
    else:
        # overnight window (e.g. 20:00–06:00)
        is_day = (tt >= start) | (tt < end)

    df[f"{prefix}is_day"]   = is_day.astype(np.int8)
    df[f"{prefix}is_night"] = (~is_day).astype(np.int8)

    # Weekday vs weekend
    dow = tindex.dayofweek  # 0=Mon,...,6=Sun
    df[f"{prefix}is_weekday"] = (dow < 5).astype(np.int8)
    df[f"{prefix}is_weekend"] = (dow >= 5).astype(np.int8)

    return df


def kernel_smooth_occupancy_counts(
    x: pd.Series,
    freq: str = "5min",
    h_minutes: int = 30,
    window_minutes: int = 60,
    lambda_days: float = 5.0,
    include_same_day: bool = True,
    a0: float = 1.0,
    b0: float = 1.0,
    name: str = "mu_hat_kernel_matrix"
) -> pd.Series:
    """
    Kernel-smoothed seasonal baseline for occupancy counts using Gamma–Poisson conjugacy.

    Parameters
    ----------
    x : pd.Series
        Occupancy counts with a DatetimeIndex (one value per slot).
    freq : str
        Slot size (e.g., '5min').
    h_minutes : int
        Gaussian kernel bandwidth across slots (time-of-day).
    window_minutes : int
        Half-window size across slots (only slots within +/- window are used).
    lambda_days : float
        Exponential decay half-life across days (in days units).
    include_same_day : bool
        If True, forbids using slots at (d0, s' >= s) so no same-day look-ahead.
        (Also excludes the current slot itself.)
    a0, b0 : float
        Gamma(a0, b0) prior on the Poisson mean (rate parameterization).
    name : str
        Name for the returned Series.

    Returns
    -------
    pd.Series
        Posterior mean baseline \hat{mu} aligned to x.index.
    """
    # Reuse your day/slot utility by passing a dummy 'n' of ones
    n_ones = pd.Series(1.0, index=pd.to_datetime(x.index))
    Y, N, days_index, times_mat, target_mask, S = make_day_slot_matrices(x, n_ones, freq=freq)
    D = Y.shape[0]

    # Valid cells: finite observations
    valid = np.isfinite(Y)

    # Convert params to slot units
    slot_minutes = pd.Timedelta(freq).seconds / 60
    h_bins = max(1, int(round(h_minutes / slot_minutes)))
    W_bins = max(1, int(round(window_minutes / slot_minutes)))

    MU = np.full((D, S), np.nan, dtype=float)

    # Precompute slot distances (circular by day)
    slot_idx = np.arange(S)

    for d in range(D):
        # Exponential decay across days (no future days)
        day_lag = (d - np.arange(D))
        w_row = np.exp(-day_lag / lambda_days)
        w_row[day_lag < 0] = 0.0
        if not include_same_day:
            w_row[day_lag == 0] = 0.0
        w_row = w_row.reshape(-1, 1)  # (D,1)

        for s in range(S):
            if not target_mask[d, s]:
                continue

            # Gaussian across slots with compact support +/- W_bins
            delta = np.abs(slot_idx - s)
            delta = np.minimum(delta, S - delta)  # wrap-around within day
            w_col = np.exp(-(delta / h_bins) ** 2)
            w_col[delta > W_bins] = 0.0
            w_col = w_col.reshape(1, -1)  # (1,S)

            W = w_row * w_col * valid  # (D,S)

            # Enforce no same-day look-ahead (also drops the current cell)
            if include_same_day:
                W[d, (s-1):] = 0.0

            if not np.any(W):
                # Prior mean under Gamma(a0,b0) with zero exposure
                MU[d, s] = a0 / b0
                continue

            Ynz = np.nan_to_num(Y, nan=0.0)
            sum_wx = np.sum(W * Ynz)
            sum_w  = np.sum(W)

            # Gamma–Poisson posterior mean
            MU[d, s] = (a0 + sum_wx) / (b0 + sum_w)

    # Map back to timestamps
    ts_to_val = {pd.Timestamp(times_mat[i, j]): MU[i, j] for i in range(D) for j in range(S)}
    idx = pd.to_datetime(x.index)
    return pd.Series([ts_to_val.get(pd.Timestamp(t), np.nan) for t in idx], index=idx, name=name)

def kernel_smoothed_occupancy_counts(
    occupancy_counts: pd.DataFrame,
    freq: str = "5min",
    h_minutes: int = 5,
    window_minutes: int = 15,
    lambda_days: float = 5.0,
    include_same_day: bool = True,
    a0: float = 1.0,
    b0: float = 1.0,
    name: str = "mu_hat_kernel_matrix"
) -> pd.DataFrame:
    """
    Apply kernel_smooth_occupancy_counts row-wise to get smoothed occupancy baselines.

    Parameters
    ----------
    occupancy_counts : DataFrame
        Rows = cells (or (origin,) if you prefer), Cols = time (DatetimeIndex).
    freq : str
        Time resolution of the series (default "5min").
    h_minutes : int
        Gaussian kernel bandwidth across slots (time-of-day).
    window_minutes : int
        Half-window size around the target slot (in minutes).
    lambda_days : float
        Exponential decay across days (in days).
    include_same_day : bool
        If True, exclude same-day slots >= current slot to avoid look-ahead.
    a0, b0 : float
        Gamma(a0, b0) prior hyperparameters (rate parameterization).
    name : str
        Base name for output series; column names are preserved from input.

    Returns
    -------
    DataFrame
        Smoothed occupancy means with same index/columns as `occupancy_counts`.
    """
    nrows, ncols = occupancy_counts.shape
    out = np.zeros((nrows, ncols), dtype=float)

    for i in range(nrows):
        x_series = occupancy_counts.iloc[i, :]

        mu_hat = kernel_smooth_occupancy_counts(
            x_series,
            freq=freq,
            h_minutes=h_minutes,
            window_minutes=window_minutes,
            lambda_days=lambda_days,
            include_same_day=include_same_day,
            a0=a0,
            b0=b0,
            name=name
        )

        out[i, :] = mu_hat.values

    return pd.DataFrame(out, index=occupancy_counts.index, columns=occupancy_counts.columns)


def stick_to_mult_matrix(pi, return_other_type = False):
    """
    Vectorized conversion from stick-breaking to multinomial probabilities.

    Parameters
    ----------
    pi : ndarray of shape (K, T)
        Stick-breaking probabilities for K gates across T time points.

    Returns
    -------
    p : ndarray of shape (K+1, T)
        Multinomial probabilities (including the 'other' bucket in the last row).
    """
    pi = np.asarray(pi, dtype=float)
    K, T = pi.shape

    # survival up to each gate: cumprod of (1 - pi) along rows
    surv = np.vstack([
        np.ones((1, T)),                     # survival before the first gate
        np.cumprod(1 - pi, axis=0)           # survival after each gate
    ])  # shape (K+1, T)

    # first K rows: p_k = pi_k * survival_{k}
    p_main = pi * surv[:-1]

    
    if return_other_type:
        # last row: "other" = survival after all K gates
        p_other = surv[-1][None, :]
        p = np.vstack([p_main, p_other])
    
        col_sums = p.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-12), \
            f"Column sums not equal to 1, got {col_sums}"

        return p
    else:
        return p_main
    
def stick_to_mult_by_group(kernel_prob_df, stick_func):
    """
    Apply stick->mult conversion per first-level group of a 2-level MultiIndex DataFrame.

    Parameters
    ----------
    kernel_prob_df : pd.DataFrame
        Rows: MultiIndex with (level0, level1). Columns: time (or any).
        Within each level0 group, rows (level1) are ordered as stick-breaking gates.
    stick_func : callable
        Function taking a (K, T) ndarray and returning a (K, T) ndarray
        (no extra 'other' row).

    Returns
    -------
    pd.DataFrame
        Same index/columns as input, with rows converted to multinomial probs.
    """
    assert isinstance(kernel_prob_df.index, pd.MultiIndex) and kernel_prob_df.index.nlevels == 2

    # apply per group on level 0; preserve order and structure
    def _apply_group(subdf):
        arr_in = subdf.to_numpy(dtype=float)         # (K, T)
        arr_out = stick_func(arr_in)                 # (K, T), no extra row
        return pd.DataFrame(arr_out, index=subdf.index, columns=subdf.columns)

    out = (kernel_prob_df
           .groupby(level=0, sort=False, group_keys=False)
           .apply(_apply_group))
    return out

def fit_binomial_dglms(
    trans_counts: pd.DataFrame,    # rows: (origin,dest), cols: time (DatetimeIndex @ 5-min ideal)
    n_counts: pd.DataFrame,        # same shape
    daily_harmonics: int = 3,
    weekly_harmonics: int = 2,
    deltrend: float = 0.98,
    delseason: float = 0.98,
    delregn: float = 0.98,
    # --- covariates ---
    pair_features_list: Optional[List[Union[pd.DataFrame, np.ndarray]]] = None,  # each (n_pairs, T)
    time_features_list: Optional[List[Union[pd.DataFrame, np.ndarray]]] = None,  # each (p_time_k, T)
):
    """
    Binomial DGLMs with flexible covariates.

    pair_features_list:
        List of per-(O,D) scalar features, each of shape (n_pairs, T).
        Example: kernel logit prob, occupancy-return * prob, etc.

    time_features_list:
        List of shared time features; each is (p_time_k, T).
        These are vertically stacked to form a (sum_k p_time_k, T) matrix.

    Returns
    -------
    models    : dict[(O,D) -> bin_dglm]
    forecasts : dict[(O,D) -> DataFrame with ['p_emp','p_hat','y','n']]
    states    : dict[(O,D) -> {'m_steps': (p_total,T), 'C_steps': (p_total,p_total,T)}]
    """
    assert trans_counts.shape == n_counts.shape, "trans_counts and n_counts must match"
    assert (trans_counts.columns == n_counts.columns).all(), "time axes must align"

    # periods for 5-min grid
    daily_period  = 288
    weekly_period = 2016
    seasPeriods = [daily_period, weekly_period]
    seasHarmComponents = [list(range(1, daily_harmonics+1)),
                          list(range(1, weekly_harmonics+1))]

    # time axis & shapes
    time_index = pd.to_datetime(trans_counts.columns)
    T = len(time_index)
    n_pairs = len(trans_counts.index)

    # ---- normalize per-pair features list ----
    norm_pair_feats: List[pd.DataFrame] = []
    if pair_features_list:
        for idx, feat in enumerate(pair_features_list):
            if isinstance(feat, np.ndarray):
                assert feat.shape == (n_pairs, T), \
                    f"pair_features_list[{idx}] must have shape {(n_pairs, T)}"
                df_f = pd.DataFrame(feat, index=trans_counts.index, columns=time_index)
            else:
                df_f = feat.copy()
                df_f.columns = time_index
                assert df_f.shape == trans_counts.shape, \
                    f"pair_features_list[{idx}] must match trans_counts shape"
                assert (df_f.index == trans_counts.index).all(), \
                    f"pair_features_list[{idx}] row index must match trans_counts.index"
            norm_pair_feats.append(df_f)
    p_pair = len(norm_pair_feats)

    # ---- normalize time features list ----
    if time_features_list:
        time_blocks = []
        for idx, tf in enumerate(time_features_list):
            if isinstance(tf, np.ndarray):
                assert tf.shape[1] == T, \
                    f"time_features_list[{idx}] must have T={T} columns"
                time_blocks.append(tf.astype(float))
            else:
                tf_df = tf.copy().reindex(columns=time_index)
                assert tf_df.shape[1] == T, \
                    f"time_features_list[{idx}] must have T={T} columns"
                time_blocks.append(tf_df.to_numpy(dtype=float))
        time_mat = np.vstack(time_blocks) if len(time_blocks) > 0 else None
        p_time = 0 if time_mat is None else time_mat.shape[0]
    else:
        time_mat = None
        p_time = 0

    # regression dimension = (#pair features) + (sum of time features)
    regn_dim = p_pair + p_time

    models: Dict[Tuple[int, int], "bin_dglm"] = {}
    forecasts: Dict[Tuple[int, int], pd.DataFrame] = {}
    states: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    # state dimension: level + 2 per harmonic + regression betas
    p_trend_season = 1 + 2 * (daily_harmonics + weekly_harmonics)
    p_total = p_trend_season + regn_dim

    # Pre-extract per-pair feature matrices for speed (list of np.ndarray with shape (n_pairs, T))
    pair_feat_mats = [df.to_numpy(dtype=float) for df in norm_pair_feats]  # length p_pair

    for row_idx, ((O, Dst), y_row) in enumerate(trans_counts.iterrows()):
        y = y_row.to_numpy(dtype=int)
        n = n_counts.loc[(O, Dst)].to_numpy(dtype=int)
        n = np.maximum(n, y)  # safety

        # model with regression
        a0 = np.zeros((p_total, 1), dtype=float)
        R0 = np.eye(p_total, dtype=float) * 10.0

        mod = bin_dglm(
            a0=a0, R0=R0,
            ntrend=1, nregn=regn_dim,
            seasPeriods=seasPeriods,
            seasHarmComponents=seasHarmComponents,
            deltrend=deltrend, delseas=delseason, delregn=delregn
        )

        # preallocate outputs
        p_emp = np.full(T, np.nan, dtype=float)
        p_hat = np.full(T, np.nan, dtype=float)

        m_steps = np.zeros((p_total, T), dtype=float)           # posterior means
        C_diag_steps = np.zeros((p_total, T), dtype=float)      # diagonals only
        C_last = np.zeros((p_total, p_total), dtype=float)      # last full covariance

        for t in range(T):
            # ---- build X_t (length = regn_dim) ----
            if regn_dim > 0:
                # per-pair scalars at time t, one from each feature
                pair_vals = [mat[row_idx, t] for mat in pair_feat_mats] if p_pair > 0 else []
                # shared time-features column at t
                time_vals = time_mat[:, t].tolist() if p_time > 0 else []
                X_t = np.array(pair_vals + time_vals, dtype=float)
            else:
                X_t = None

            # ---- one-step-ahead probability forecast (mean_only) ----
            n_use = 1
            if regn_dim > 0:
                fc_mean = mod.forecast_marginal(k=1, n=n_use, X=X_t, mean_only=True)
            else:
                fc_mean = mod.forecast_marginal(k=1, n=n_use, mean_only=True)
            p_hat[t] = fc_mean / n_use

            # ---- update with observation ----
            if n[t] > 0:
                p_emp[t] = y[t] / n[t]
                if regn_dim > 0:
                    mod.update(y=int(y[t]), n=int(n[t]), X=X_t)
                else:
                    mod.update(y=int(y[t]), n=int(n[t]))
            else:
                p_emp[t] = np.nan
                # skip update

            # ---- save posterior ----
            m_steps[:, t] = np.asarray(mod.a, dtype=float).ravel()
            R_now = np.asarray(mod.R, dtype=float)
            C_diag_steps[:, t] = np.diag(R_now)
            if t == T - 1:
                C_last = R_now.copy()

        models[(O, Dst)] = mod
        forecasts[(O, Dst)] = pd.DataFrame(
            {"p_emp": p_emp, "p_hat": p_hat, "y": y.astype(float), "n": n.astype(float)},
            index=time_index
        )
        states[(O, Dst)] = {"m_steps": m_steps, "mVar_steps": C_diag_steps, "C_last": C_last}

    return models, forecasts, states

################################## Utilities for Individual-level Model ########################################


def onehot_to_cascade(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each level(0) index group and each column,
    set all values below the first 1 to NaN.
    Columns with no 1 remain unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex rows (level 0 for grouping), columns are time/features.

    Returns
    -------
    pd.DataFrame
        Masked DataFrame.
    """
    out = df.copy()

    for lvl0, sub_df in df.groupby(level=0):
        # lvl0 = 30; sub_df = transition_counts_ind.loc[[30]]
        
        notna_idx = np.where(~sub_df.isna().all(axis=0))[0]
        
        arr = sub_df.iloc[:, notna_idx].to_numpy()
        first_one_idx = np.argmax(arr == 1, axis=0)

        row_idx = np.arange(arr.shape[0])[:, None]
        # only mask columns wherenotna_idx there IS a 1
        mask = (row_idx > first_one_idx[None, :])
        arr[mask] = np.nan

        sub_df.values[:, notna_idx] = arr
        out.loc[lvl0] = sub_df

    return out


def multinomial_to_cascade(P: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Convert column-wise multinomial probabilities (K x T) into stick-breaking probs (K x T).
    Rows define the stick order: v_k = p_k / (1 - sum_{j<k} p_j).

    Parameters
    ----------
    P : pd.DataFrame
        Shape (K, T). Each column is a multinomial over K categories (may sum to <= 1).
    eps : float
        Numerical floor to avoid division-by-zero / negatives in the remaining mass.

    Returns
    -------
    V : pd.DataFrame
        Shape (K, T). Stick-breaking probabilities v_1..v_K.
        For full multinomials (column sums = 1), the last row will be exactly 1.
    """
    arr = P.to_numpy(dtype=float)                # (K, T)
    K, T = arr.shape

    # cumulative sum down rows: cum[k] = sum_{i<=k} p_i
    cum = np.cumsum(arr, axis=0)                 # (K, T)

    # prev sums: sum_{i<k} p_i  (same shape as arr)
    prev = np.vstack([np.zeros((1, T)), cum[:-1, :]])

    # remaining mass before taking p_k
    rem = 1.0 - prev                             # (K, T)
    rem = np.clip(rem, eps, None)                # guard tiny/negative due to rounding

    # stick-breaking probs
    V = arr / rem                                # (K, T)

    # Preserve NaNs from inputs
    V[~np.isfinite(arr)] = np.nan

    # Clip to [0, 1] just in case of tiny numerical overshoots
    V = np.clip(V, 0.0, 1.0)

    return pd.DataFrame(V, index=P.index, columns=P.columns)


def cascade_to_multinomial(V: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Convert stick-breaking probabilities (K x T) back to multinomial probabilities (K x T).

    Given v_k (k = 1..K) in stick order, the multinomial probabilities are:
        p_k = v_k * prod_{j<k} (1 - v_j)

    Notes
    -----
    - If the original multinomial columns summed to 1, then v_K = 1 and the
      reconstruction exactly recovers P.
    - If columns sum to < 1 (partial multinomial), the resulting P will
      sum to the same total mass encoded by V.
    - NaNs in V propagate to P for the same cells.

    Parameters
    ----------
    V : pd.DataFrame
        Shape (K, T). Stick-breaking probabilities v_1..v_K per column, in stick order.
    eps : float
        Numerical floor for clipping (used only for safety).

    Returns
    -------
    pd.DataFrame
        P (K x T): multinomial probabilities with the same index/columns as V.
    """
    v = V.to_numpy(dtype=float)                 # (K, T)
    K, T = v.shape

    # remaining mass before taking k is prod_{j<k} (1 - v_j)
    # Build it via cumulative product:
    # rem_prefix[0,:] = 1
    # rem_prefix[k,:] = prod_{j<k}(1 - v_j)
    one_minus_v = 1.0 - v
    rem_prefix = np.vstack([np.ones((1, T), dtype=float),
                            np.cumprod(one_minus_v[:-1, :], axis=0)])

    # multinomial probs
    P = v * rem_prefix

    # Preserve NaNs from inputs (stick-breaking cells that are NaN → P NaN)
    P[~np.isfinite(v)] = np.nan

    # Clip tiny numerical artifacts
    P = np.clip(P, 0.0, 1.0)

    return pd.DataFrame(P, index=V.index, columns=V.columns)


def build_Xy_summary(
    input_transition_ind: pd.DataFrame,
    X_array: np.ndarray,
    feat_cols: list, 
    max_streak: int = 6,
    logit_clip: float = 10.0,
    logit_cols: list[str] = None
) -> pd.DataFrame:
    """
    Build Xy_summary for staying-to-self modeling.

    Parameters
    ----------
    input_transition_ind : pd.DataFrame
        K x T matrix of {0,1,NaN} indicators (rows=OD_idx, cols=time_idx).
    X_array : np.ndarray
        Shape (F, K, T). Feature stack where:
          [0] = kernel_prob, [1..F-2] = lag features, [F-1] = leave_gate.
    max_streak : int
        Streak limit passed to drop_beyond_streak_fn when provided.
    logit_clip : float
        Clip logit outputs to [-logit_clip, logit_clip].
    logit_cols : list of str
        Explicit list of column names to apply logit transformation to.
        If None, defaults to ['kernel_prob', lag columns].

    Returns
    -------
    pd.DataFrame
        Xy_summary with columns:
          ['OD_idx','time_idx','y','kernel_prob','lag1'..,'lagN','leave_gate','weight']
    """
    # -- gather non-NaN indices sorted by time --
    nz = np.where(input_transition_ind.notna())
    order = np.argsort(nz[1])
    od_idx = nz[0][order]
    t_idx = nz[1][order]

    non_zero_idx = pd.DataFrame(
        {'OD_idx': od_idx, 'time_idx': t_idx},
        index=np.arange(len(od_idx))
    )
    non_zero_idx['y'] = input_transition_ind.values[od_idx, t_idx].astype(int)

    # -- feature names --
    # F = X_array.shape[0]
    # feat_cols = (['kernel_prob'] +
    #              [f'lag{k+1}' for k in range(n_lags)] +
    #              ['leave_gate'])

    # -- select feature values --
    X_sel = X_array[:, od_idx, t_idx].T
    X_df = pd.DataFrame(X_sel, columns=feat_cols, index=non_zero_idx.index)

    Xy_summary = pd.concat([non_zero_idx, X_df], axis=1)

    # -- logit transformation with clipping --
    eps = 1e-9
    Xy_summary[logit_cols] = np.clip(Xy_summary[logit_cols], eps, 1 - eps)
    Xy_summary[logit_cols] = scipy.special.logit(Xy_summary[logit_cols])
    Xy_summary[logit_cols] = Xy_summary[logit_cols].clip(-logit_clip, logit_clip)

    # -- streak pruning --
    Xy_summary = drop_beyond_streak(
        Xy_summary, group_col='OD_idx', y_col='y', max_streak=max_streak
    )

    # -- compute weights --
    cum_mean = Xy_summary['y'].cumsum() / np.arange(1, len(Xy_summary) + 1)
    Xy_summary['weight'] = ((1 - cum_mean) / cum_mean).shift(1).replace([np.inf, -np.inf], np.nan)

    return Xy_summary




def fill_forecasted_probabilities(
    input_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    input_transition_ind: pd.DataFrame,
    od_col: str = "OD_idx",
    time_col: str = "time_idx",
    p_hat_col: str = "p_hat"
) -> pd.DataFrame:
    """
    Merge forecast probabilities with transition indicator matrix
    and fill missing values forward along time for each OD pair.

    Parameters
    ----------
    input_df : pd.DataFrame
        Contains at least [od_col, time_col].
    forecasts : pd.DataFrame
        Contains forecast results with index as time or a time column, and column p_hat_col.
    input_transition_ind : pd.DataFrame
        K x T transition indicator matrix (0/1/NaN) with OD_idx as rows and time_idx as columns.
    od_col : str, default "OD_idx"
        Column name for OD index in input_df and forecasts.
    time_col : str, default "time_idx"
        Column name for time index in input_df and forecasts.
    p_hat_col : str, default "p_hat"
        Column name for forecasted probabilities in forecasts.

    Returns
    -------
    forecasted_prob_df : pd.DataFrame
        A K x T matrix of forecasted probabilities,
        forward filled along time within each OD pair and masked by input_transition_ind.
    """
    # merge forecast results with OD/time indices
    # forecasts_df = pd.merge(
    #     input_df[[od_col, time_col]],
    #     forecasts.reset_index().rename(columns={"index": time_col})
    # )
    input_df['p_hat'] = forecasts['p_hat'].values
    forecasts_df = input_df[[od_col, time_col, 'p_hat']]

    # initialize result matrix with NaNs
    forecasted_prob_df = input_transition_ind * np.nan

    # assign probabilities at corresponding OD/time positions
    forecasted_prob_df.values[
        forecasts_df[od_col],
        forecasts_df[time_col]
    ] = forecasts_df[p_hat_col].values

    # forward fill along time axis (per OD row)
    forecasted_prob_df = forecasted_prob_df.ffill(axis=1)

    # mask with original transition indicator
    # forecasted_prob_df[input_transition_ind.isna()] = np.nan

    return forecasted_prob_df


def classify_zones_by_stay(series: pd.Series, min_stay_blocks: int = 6) -> dict:
    """
    Classify zones as staying or passing based on consecutive stay duration.
    
    Parameters
    ----------
    series : pd.Series
        Time series of zone indices (integers).
    min_stay_blocks : int
        Minimum number of consecutive 5-min intervals to qualify as staying zone.
    
    Returns
    -------
    dict
        {
            'staying_zones': [(zone, proportion), ...],
            'passing_zones': [(zone, proportion), ...]
        }
    """
    # Step 1: Identify consecutive segments
    # (shift !=) creates a new segment whenever the zone changes
    group_ids = (series != series.shift()).cumsum()
    segments = series.groupby(group_ids).agg(['first', 'size'])
    
    # Step 2: Find zones that have at least one segment >= min_stay_blocks
    staying_candidates = set(segments.loc[segments['size'] >= min_stay_blocks, 'first'])
    
    # Step 3: Calculate zone proportions
    proportions = series.value_counts(normalize=True).sort_values(ascending=False)
    
    # Step 4: Separate into staying and passing
    staying = [zone for zone, prop in proportions.items() if zone in staying_candidates]
    passing = [zone for zone, prop in proportions.items() if zone not in staying_candidates]
    
    return {
        'staying_zones': staying,
        'passing_zones': passing
    }


def make_individual_transition_matrix(agent_trajectory: pd.Series) -> pd.DataFrame:
    """
    Construct an individual transition indicator matrix (Origin, Destination) × Time
    for a single agent trajectory.

    For each time step t where a transition occurs:
      - the row corresponding to the true (origin, destination) pair gets 1
      - other rows with the same origin get 0
      - all other rows with different origin get NaN
      
    Parameters
    ----------
    agent_trajectory : pd.Series
        A time-indexed Series containing zone/location IDs (origins/destinations).
        - Index: timestamp or time-step when the agent is at a location
        - Values: location (zone) IDs
        
    Returns
    -------
    pd.DataFrame
        A transition matrix of shape (#OD_pairs, #times), with:
        - Rows indexed by (origin, destination) MultiIndex
        - Columns = destination time steps (pd.DatetimeIndex)
        - Values:
            * 1 → transition from origin to destination at this time
            * 0 → origin was active but transitioned to a different destination
            * NaN → no transition involving this origin at this time

    """
    
    aloc = agent_trajectory.dropna()
    
    
    origins = aloc.values [:-1]
    dests = aloc.values[1:]
    times = pd.to_datetime(aloc.index[1:]) # transitions at destination time t[1:]
    
    # create full origin-destination index
    all_pairs = pd.Series(tuple(zip(origins,dests))).unique()
    full_pairs = pd.MultiIndex.from_tuples(all_pairs, names=["origin", "dest"])
    # initialize output with NaN
    tc = pd. DataFrame( index=full_pairs, columns=times, dtype=float)
    
    # fill in 0s and 1s
    for o, d, t in zip(origins, dests, times):
        mask = (tc.index.get_level_values("origin") == o)
        tc.loc[mask, t] = int(0)
        tc.loc[(o, d), t] = int(1)
    return tc



def get_transition_indicator(traj_full, threshold=10, end_of_trn = 8063):
    # traj_full = agent_i_full
    trans_ind_full = make_individual_transition_matrix(traj_full)
    trans_ind_full = onehot_to_cascade(trans_ind_full)
    
    direction_counts = trans_ind_full.iloc[:, :end_of_trn].sum(axis=1)
    valid_pair = trans_ind_full.index[direction_counts > threshold]
    
    return trans_ind_full.loc[valid_pair], trans_ind_full.drop(index=valid_pair)




def split_transitions_by_zone_type(transition_counts_ind: pd.DataFrame, zone_types) -> dict:
    """
    Split K×T indicator DF (index=['origin','dest']) into:
      1) staying→staying
      2) staying→any
      3) passing→any
    Returns a dict of DataFrames with the same columns (time).
    """

    # ---- normalize zone type inputs into sets ----
    if isinstance(zone_types, dict) and 'staying_zones' in zone_types and 'passing_zones' in zone_types:
        staying_set  = set(zone_types['staying_zones'])
        passing_set  = set(zone_types['passing_zones'])
    elif isinstance(zone_types, pd.Series):
        # 0 = staying, 1 = passing
        staying_set  = set(zone_types[zone_types == 0].index)
        passing_set  = set(zone_types[zone_types == 1].index)
    else:
        raise ValueError("zone_types must be a dict with 'staying_zones'/'passing_zones' or an indicator pd.Series (0=staying,1=passing).")

    # ---- sanity check index names ----
    if list(transition_counts_ind.index.names) != ['origin', 'dest']:
        raise ValueError("transition_counts_ind index must be a MultiIndex with names ['origin','dest'].")

    # ---- build masks ----
    origin = transition_counts_ind.index.get_level_values('origin')
    dest   = transition_counts_ind.index.get_level_values('dest')

    mask_stay_stay = origin.isin(staying_set) & (origin == dest)
    mask_stay_any  = origin.isin(staying_set) & (origin != dest)
    mask_pass_any  = origin.isin(passing_set) 

    # ---- slice and return ----
    return {
        'staying_to_self': transition_counts_ind.loc[mask_stay_stay],
        'staying_to_other':     transition_counts_ind.loc[mask_stay_any],
        'passing_to_any':     transition_counts_ind.loc[mask_pass_any],
    }


def drop_beyond_streak(df, group_col='OD_idx', y_col='y', max_streak=2):
    '''
    Drop consecutive 1/0 to balance the proportion of 1/0. Otherwise model will be tilted towards one category.
    '''
    
    # within each group, identify runs of equal y via change points
    def _mask_group(g):
        run_id = g[y_col].ne(g[y_col].shift()).cumsum()
        run_pos = g.groupby(run_id).cumcount()  # 0,1,2,... within each run
        return run_pos >= max_streak            # True => drop

    drop_mask = (
        df.sort_values([group_col, 'time_idx'])  # ensure time order
          .groupby(group_col, group_keys=False)
          .apply(_mask_group)
    )
    # Keep only those not marked to drop
    return df.loc[~drop_mask].copy()

def fit_individual_bernoulli_dglm(
    xy: pd.DataFrame,                      # columns: time_idx, y, kernel_prob, lag1..lag6
    time_col: str = "time_idx",
    y_col: str = "y",
    feature_cols: Optional[List[str]] = None,   # default: ['kernel_prob'] + all 'lag*'
    daily_harmonics: int = 0,
    weekly_harmonics: int = 0,
    deltrend: float = 1,
    delseason: float = 1,
    delregn: float = 1,
    include_intercept: bool = True,
    upweight: bool  = True,
    upweight_type: int = 1,
    pr_mean: np.array = None,
    pr_var: float = 50.0
):
    """
    Fit a Bernoulli DGLM (via binomial DGLM with n=1) for individual-level binary events.

    This version processes observations row by row, allows multiple updates
    per time index (without evolution between observations at the same time),
    and supports optional *upweighting* for rare classes.

    Parameters
    ----------
    xy : pd.DataFrame
        Input data. Must contain:
        - time_col (e.g., 'time_idx'): time step or group identifier
        - y_col (e.g., 'y'): binary response {0, 1}
        - feature columns: regressors such as 'kernel_prob', 'lag1'...'lag6'
        - if upweight=True: a 'weight' column for upweighting.
    time_col : str, default="time_idx"
        Name of the column containing the time index. Observations with the same
        time index are treated as simultaneous events (multiple updates at same time).
    y_col : str, default="y"
        Name of the column containing binary target variable {0,1}.
    feature_cols : list of str, optional
        List of feature column names. If None, defaults to
        ['kernel_prob'] plus all columns starting with 'lag' (sorted).
    daily_harmonics : int, default=0
        Number of daily harmonic seasonal components (0 disables daily seasonality).
    weekly_harmonics : int, default=0
        Number of weekly harmonic seasonal components (0 disables weekly seasonality).
    deltrend : float, default=1
        Discount factor for the trend component (1 means no evolution).
    delseason : float, default=1
        Discount factor for seasonal components.
    delregn : float, default=1
        Discount factor for regression coefficients.
    include_intercept : bool, default=True
        Whether to include an intercept term in the regression design vector X_t.
    upweight : bool, default=True
        If True, will upweight observations whose response equals `upweight_type`
        by applying extra update steps proportional to `xy['weight'] - 1`.
    upweight_type : int, default=1
        Target value (e.g., 0 or 1) to upweight if `upweight` is True.

    Returns
    -------
    mod : pybats.dglm.bin_dglm
        The fitted Bernoulli/binomial DGLM model object.
    forecasts : pd.DataFrame
        A DataFrame indexed by time, containing:
        - 'p_hat': one-step-ahead forecast probability for each observation
        - 'y': observed binary outcome.
    states : dict
        Posterior state summaries with:
        - 'm_steps': state mean vectors over time, shape (p_total, T)
        - 'mVar_steps': posterior variances (diagonal of R), shape (p_total, T)
        - 'C_last': posterior covariance at final step.

    Notes
    -----
    - Observations with the same time index are updated sequentially with
      delta = 1.0 to freeze evolution.
    - Evolution occurs once per *unique* time index (first update of each group).
    - `upweight` effectively gives more influence to rare events by repeating updates.
    - Feature values are logit-transformed and clipped to [-10, 10] for stability.
    """
    # xy = input_df; feature_cols = ['kernel_prob', 'lag1', 'lag2']
    # daily_harmonics = 0; weekly_harmonics = 0
    # deltrend = 1; delseason = 1; delregn = 1
    # include_intercept = True
    # time_col = "time_idx";
    # y_col = "y"
    # include_intercept = True; upweight = False; upweight_type=1
    # pr_mean = None; pr_var = 50
    # --- order by time
    # xy = xy.sort_values(by=time_col).reset_index(drop=True)

    # --- pick features
    if feature_cols is None:
        base = ['kernel_prob'] if 'kernel_prob' in xy.columns else []
        lags = sorted([c for c in xy.columns if c.lower().startswith('lag')])
        feature_cols = base + lags
    X_mat = xy[feature_cols].to_numpy(dtype=float).T  # (p_reg, T)
    regn_dim = X_mat.shape[0]
    # if include_intercept:
    #     X_mat = np.vstack([np.ones((1, X_mat.shape[1])), X_mat])

    # --- response & time axis
    y = xy[y_col].to_numpy(dtype=float)
    times = xy[time_col].to_numpy()
    T = len(y)

    # --- seasonal setup
    # periods for 5-min grid
    daily_period  = 288
    weekly_period = 2016
    seasPeriods = [daily_period, weekly_period]
    seasHarmComponents = [list(range(1, daily_harmonics+1)),
                          list(range(1, weekly_harmonics+1))]

    # --- state dimension
    p_trend_season = 1 + 2 * (daily_harmonics + weekly_harmonics)
    p_total = p_trend_season + regn_dim

    # --- init model
    if pr_mean is None:
        a0 = np.zeros((p_total, 1))
    else:
        a0 = pr_mean
    R0 = np.eye(p_total) * pr_var
    mod = bin_dglm(
        a0=a0, R0=R0,
        ntrend=int(include_intercept), nregn=regn_dim,
        seasPeriods=seasPeriods,
        seasHarmComponents=seasHarmComponents,
        deltrend=deltrend, delseas=delseason, delregn=delregn
    )

    # outputs
    p_hat = np.full(T, np.nan)
    m_steps = np.zeros((p_total, T))
    C_diag_steps = np.zeros((p_total, T))
    C_last = np.zeros((p_total, p_total))

    # main loop: multiple updates per time_idx, no evolution between them
    for t in range(T):
        # t = 11969
        X_t = X_mat[:, t] if regn_dim > 0 else None
        # X_t = np.clip(scipy.special.logit(X_t), -10, 10)
        
        # one-step-ahead Bernoulli forecast (from current prior at this time)
        if regn_dim > 0:
            p_hat[t] = mod.forecast_marginal(k=1, n=1, X=X_t, mean_only=True)
        else:
            p_hat[t] = mod.forecast_marginal(k=1, n=1, mean_only=True)
        
        if not np.isnan(y[t]) and xy['update'].iloc[t]:
        # update with this observation (no evolution drift if same_time due to deltas=1)
            if regn_dim > 0:
                mod.update(y=int(y[t]), n = 1, X=X_t)
            else:
                mod.update(y=int(y[t]), n = 1)
            
            # additional update for the much less category 
            if y[t] == upweight_type and upweight and xy['weight'].iloc[t] > 1:
                if regn_dim > 0:
                    mod.update(y=int(y[t]), n = (xy['weight'].iloc[t] - 1), X=X_t)
                else:
                    mod.update(y=int(y[t]), n = (xy['weight'].iloc[t] - 1))
    
        # store posterior
        m_steps[:, t] = np.asarray(mod.a).ravel()
        R_now = np.asarray(mod.R)
        C_diag_steps[:, t] = np.diag(R_now)
        C_last = R_now  # keep last

    forecasts = pd.DataFrame({"p_hat": p_hat, "y": y.astype(float)}, index=times)
    states = {"m_steps": m_steps, "mVar_steps": C_diag_steps, "C_last": C_last}
    return mod, forecasts, states


def continue_individual_bernoulli_dglm(
    mod,                           # existing pybats.dglm.bin_dglm object (mutated in-place)
    xy: pd.DataFrame,              # columns: time_idx, y, features..., and 'update' (bool)
    time_col: str = "time_idx",
    y_col: str = "y",
    feature_cols: Optional[List[str]] = None,   # default: ['kernel_prob'] + all 'lag*'
    upweight: bool = True,
    upweight_type: int = 1
) -> Tuple[object, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Continue forecasting & updating a pre-existing Bernoulli/binomial DGLM (pybats).
    - No model construction here; `mod` is required and will be updated in-place.
    - Multiple rows per time index are processed sequentially without extra evolution.

    Parameters
    ----------
    mod : pybats.dglm.bin_dglm
        Existing model; will be mutated.
    xy : pd.DataFrame
        Must contain:
          - time_col (e.g., 'time_idx')
          - y_col (binary 0/1; NaN allowed to skip update)
          - feature columns (for regression X), if any
          - 'update' boolean column to gate updates
          - optional 'weight' if upweight=True
    feature_cols : list[str] or None
        If None: ['kernel_prob'] + all columns starting with 'lag' (sorted).

    Returns
    -------
    mod : the same (mutated) model object
    forecasts : pd.DataFrame (index = time)
        - 'p_hat': one-step-ahead forecast probability
        - 'y': observed outcome
    states : dict
        - 'm_steps': state mean vectors over time (shape [p_total, T])
        - 'mVar_steps': posterior variance diagonal per step (shape [p_total, T])
        - 'C_last': final posterior covariance matrix
    """
    # --- choose features
    if feature_cols is None:
        base = ['kernel_prob'] if 'kernel_prob' in xy.columns else []
        lags = sorted([c for c in xy.columns if c.lower().startswith('lag')])
        feature_cols = base + lags

    X_mat = xy[feature_cols].to_numpy(dtype=float).T if feature_cols else None  # (p_reg, T) or None
    regn_dim = 0 if X_mat is None else X_mat.shape[0]

    # --- response & index
    y = xy[y_col].to_numpy(dtype=float)
    times = xy[time_col].to_numpy()
    T = len(y)

    # --- storage (infer state size from model)
    p_total = int(np.asarray(mod.a).shape[0])
    p_hat = np.full(T, np.nan)
    m_steps = np.zeros((p_total, T))
    C_diag_steps = np.zeros((p_total, T))
    C_last = np.asarray(mod.R).copy()

    # --- main pass
    for t in range(T):
        X_t = (X_mat[:, t] if regn_dim > 0 else None)

        # one-step-ahead forecast
        if regn_dim > 0:
            p_hat[t] = mod.forecast_marginal(k=1, n=1, X=X_t, mean_only=True)
        else:
            p_hat[t] = mod.forecast_marginal(k=1, n=1, mean_only=True)

        # conditional update
        if not np.isnan(y[t]) and bool(xy['update'].iloc[t]):
            if regn_dim > 0:
                mod.update(y=int(y[t]), n=1, X=X_t)
            else:
                mod.update(y=int(y[t]), n=1)

            # optional upweight (repeat additional updates)
            if upweight and (y[t] == upweight_type) and ('weight' in xy.columns):
                extra = int(max(0, xy['weight'].iloc[t] - 1))
                if extra > 0:
                    if regn_dim > 0:
                        mod.update(y=int(y[t]), n=extra, X=X_t)
                    else:
                        mod.update(y=int(y[t]), n=extra)

        # store posterior (after this step's updates)
        m_steps[:, t] = np.asarray(mod.a).ravel()
        R_now = np.asarray(mod.R)
        C_diag_steps[:, t] = np.diag(R_now)
        C_last = R_now

    forecasts = pd.DataFrame({"p_hat": p_hat, "y": y.astype(float)}, index=times)
    states = {"m_steps": m_steps, "mVar_steps": C_diag_steps, "C_last": C_last}
    return mod, forecasts, states


def summarize_binary_predictions(
    forecasts: pd.DataFrame,
    states: dict = None,
    threshold: float = 0.5,
    plot_coeff: bool = False,
    verbose: bool = False
):
    """
    Quick summary of binary classification predictions based on forecast probs.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Must contain columns ['y', 'p_hat'].
    states : dict, optional
        Posterior state dict from DGLM (with 'm_steps' and 'mVar_steps').
        If provided, prints t-stats and optionally plots coefficient trajectories.
    threshold : float, default=0.5
        Threshold for converting probabilities into binary predictions.
    plot_coeff : bool, default=True
        Whether to plot coefficient trajectories if `states` is provided.

    Returns
    -------
    results_df : pd.DataFrame
        One-row DataFrame containing:
        - confusion matrix elements (TN, FP, FN, TP)
        - accuracy, precision, recall, F1, balanced accuracy, MCC
        - t-stats (as separate columns t_0, t_1, ..., if available)
    """
    # --- classification metrics ---
    y_true = forecasts['y'].values
    y_pred = (forecasts['p_hat'].values >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    if verbose:
        print("\n=== Confusion Matrix ===")
        print(cm)
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, digits=4))
    
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"MCC: {mcc:.4f}")

    # --- build results dictionary ---
    results = {
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'balanced_accuracy': bal_acc,
        'mcc': mcc
    }

    # --- coefficient t-stats ---
    if states is not None and 'm_steps' in states and 'mVar_steps' in states:
        last_mean = states['m_steps'][:, -1]
        last_var = states['mVar_steps'][:, -1]
        t_stats = last_mean / np.sqrt(last_var)
        if verbose:
            print("\n=== Final Coefficient t-stats ===")
            print(t_stats)

        # Add each t-stat as a column
        for i, t_val in enumerate(t_stats):
            results[f"t_{i}"] = t_val

        if plot_coeff:
            plt.figure(figsize=(10, 4))
            plt.plot(states['m_steps'].T)
            plt.title("Coefficient Trajectories")
            plt.xlabel("Time")
            plt.ylabel("Coefficient Value")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    # --- convert to DataFrame ---
    results_df = pd.DataFrame([results])
    return results_df


def assemble_multinomial_with_rare(
    forecasted_prob_df: pd.DataFrame,
    rare_directions_cascade: pd.DataFrame,
    cascade_to_multinomial_func
) -> pd.DataFrame:
    """
    For each origin O (level-0 of the MultiIndex rows in forecasted_prob_df):
      1) Convert cascade/stick-breaking probs -> multinomial probs via cascade_to_multinomial_func.
      2) Compute residual mass per time (1 - sum over modeled rows) on columns where all modeled rows are non-NaN.
      3) Spread the residual evenly across the 'rare' rows for that origin (from rare_directions_cascade).
      4) Concatenate modeled multinomial rows with the rare rows.

    Parameters
    ----------
    forecasted_prob_df : pd.DataFrame
        MultiIndex rows (level 0 = origin, level 1 = dest), columns = times.
        Entries are cascade (stick-breaking) probabilities for the modeled directions.
    rare_directions_cascade : pd.DataFrame
        MultiIndex with the same (origin,dest) structure for rare directions (same columns).
        Only its row index and columns are used for shape; values are ignored.
    cascade_to_multinomial_func : callable
        Function that maps a KxT DataFrame of cascade probs -> KxT multinomial probs
        (must preserve index and columns).

    Returns
    -------
    pd.DataFrame
        Big DataFrame: for all origins, the modeled multinomial probs plus the rare rows
        with residual mass split evenly (per time).
    """
    # Ensure column alignment between the two inputs
    common_cols = forecasted_prob_df.columns.intersection(rare_directions_cascade.columns)
    modeled = forecasted_prob_df.reindex(columns=common_cols)
    rare_tpl = rare_directions_cascade.reindex(columns=common_cols)

    out_blocks = []

    for O in modeled.index.get_level_values(0).unique():
        # 1) convert cascade -> multinomial for this origin
        block_cascade = modeled.loc[[O]]
        multi_prob_i = cascade_to_multinomial_func(block_cascade)

        # 2) make rare block shell (same columns, rare rows for this origin)
        if O in rare_tpl.index.get_level_values(0):
            rare_i = rare_tpl.loc[[O]].copy()
        else:
            # no rare rows for this origin
            rare_i = None

        if rare_i is not None and not rare_i.empty:
            # valid times = columns where all modeled (non-rare) rows are available (no NaNs)
            valid_time_idx = np.where(multi_prob_i.notna().all(axis=0))[0]

            # residual mass per time (NaN where invalid)
            resid_prob = pd.Series(np.nan, index=range(rare_i.shape[1]))
            resid_prob.iloc[valid_time_idx] = 1.0 - multi_prob_i.iloc[:, valid_time_idx].sum(axis=0).values

            # 3) spread residual evenly across rare rows (row-wise broadcast)
            #    rare_i has R rows -> each gets resid_prob / R at valid times
            split = (resid_prob / max(rare_i.shape[0], 1)).values
            rare_i[:] = split  # broadcast to all rows

            # 4) concat modeled + rare
            block_full = pd.concat([multi_prob_i, rare_i], axis=0)
        else:
            block_full = multi_prob_i

        out_blocks.append(block_full)

    # Row-combine all origins
    big_df = pd.concat(out_blocks, axis=0)
    return big_df


def predict_argmax_dest_and_accuracy(
    multi_all: pd.DataFrame,
    agent_tmp: pd.Series,
    start_cols: int = 288*7,
    step_minutes: int = 5,
):
    """
    From multinomial prob table (rows: (origin,dest), cols: timestamps) and a trajectory agent_tmp,
    for each time t (starting at column offset `start_cols`), pick the argmax destination D among
    rows (origin=O_t, dest=*) and evaluate accuracy vs the realized next-step dest.

    Returns
    -------
    top_dest : pd.DataFrame
        Columns: ['origin','time_idx','dest','prob','real_dest'], sorted by time_idx.
    accuracy : float
        Accuracy of predicted dest vs realized dest (dropping rows with missing ground truth).
    origins_all_nan : list
        Origins whose probability rows are all NaN across the evaluated time window
        (these origins are effectively dropped).
    """
    # multi_all = multi_all_oam; agent_tmp = agent_i_full; start_cols = 288*7; step_minutes = 5
    # slice the time window
    cols_slice = multi_all.columns[start_cols:]
    probs_block = multi_all.loc[:, cols_slice]

    # compute origins that have all-NaN probs across all dest rows in the window
    row_all_nan = probs_block.isna().all(axis=1)                          # per (origin,dest)
    origin_all_nan = row_all_nan.groupby(level=0).transform('all')        # tag all rows of such origins
    origins_all_nan = probs_block.index.get_level_values(0)[origin_all_nan].unique().tolist()

    # long format: (origin, dest, time_idx, prob)
    long = (
        probs_block
          .stack(future_stack=True)      # stack columns (timestamps) into index level
          .to_frame('prob')
          .reset_index()
          .rename(columns={'level_2': 'time_idx'})
    )

    # set (origin, time_idx) as index for subselection
    long = long.set_index(['origin', 'time_idx'])

    # build the (origin,time_idx) pairs from the trajectory:
    # origins at times t = start_cols .. end-1, predict for next step, so shift index by +step_minutes
    origins = agent_tmp.iloc[start_cols:-1].copy()
    origins.index = origins.index + pd.Timedelta(minutes=step_minutes)

    target_idx = pd.MultiIndex.from_arrays(
        [origins.values, origins.index],
        names=['origin','time_idx']
    )

    # subset to those (origin,time_idx) pairs (missing pairs drop out)
    long_sub = long.loc[target_idx].copy()

    # make 'dest' part of the index: (origin, time_idx, dest)
    df_idx = long_sub.set_index('dest', append=True)

    # drop rows where prob is NaN (e.g. unmodeled rare zones)
    df_idx = df_idx.dropna(subset=['prob'])

    # for each (origin, time_idx), choose dest with max prob
    if len(df_idx) == 0:
        # nothing to evaluate
        empty = pd.DataFrame(columns=['origin','time_idx','dest','prob','real_dest'])
        return empty, np.nan, origins_all_nan

    idx_max = df_idx.groupby(level=[0, 1])['prob'].idxmax()
    top_dest = df_idx.loc[idx_max].sort_index().reset_index().sort_values('time_idx')

    # realized next-step destination at those time_idx
    # (safe alignment via reindex; returns NaN for missing)
    top_dest['real_dest'] = agent_tmp.reindex(top_dest['time_idx']).values
    
    assert not top_dest['prob'].isna().any(), 'NaN in forecasted probabilities, please check!'
    return top_dest

def append_forecasted_rare_transitions(
    fitted_df: pd.DataFrame,            # formerly: multi_all_oam
    rare_cascade: pd.DataFrame,         # formerly: rare_directions_cascade
    trans_cube: np.ndarray,             # formerly: K_step_trans_mat[1], shape (I, J, T)
) -> pd.DataFrame:
    """
    Append forecasted group-level transition probability of rare transitions to the df of fitted directions.
    
    We need to do this since there are some rare transitions whose origin are not modeled. In this case, 
    we directly impute with forecasted group-level transition probabilities. Note that we currently assume
    that the transition array obtained from training period does not change in testing, should be changed later!

    Parameters
    ----------
    fitted_df : DataFrame
        The already-fitted directions matrix, indexed by MultiIndex (origin, dest), columns = time axis.
    rare_cascade : DataFrame
        Rare directions in cascade format, indexed by MultiIndex (origin, dest), columns = time axis.
    trans_cube : ndarray
        3D transition probability tensor with shape (n_origin, n_dest, n_time).
        We assume indices in rare_cascade (origin, dest) are integer-coded so they can index into this cube.

    Returns
    -------
    DataFrame
        Concatenation of fitted_df and the filled rare transitions (aligned on index; columns aligned by union).
    """

    # 1) Determine which origins are not modeled in `fitted_df` but appear in `rare_cascade`.
    all_origins = pd.Index(rare_cascade.index.get_level_values(0).unique())
    modeled_origins = pd.Index(fitted_df.index.get_level_values(0).unique())
    unmodeled_origins = all_origins.difference(modeled_origins)

    # Fast exit: nothing to append
    if len(unmodeled_origins) == 0:
        return fitted_df.copy()

    # 2) Build a mask of (origin,dest,time) where rare_cascade has observed cells for unmodeled origins.
    sub = rare_cascade[rare_cascade.index.get_level_values(0).isin(unmodeled_origins)]
    mask = sub.notna()

    # If everything is NaN, returns fitted_df
    if not mask.to_numpy().any():
        return fitted_df.copy()

    # 3) Find positions (row_pos, col_pos) where mask is True.
    row_pos, col_pos = np.where(mask)

    # 4) Convert row positions → actual (origin, dest) labels; pack to a 2D array.
    row_labels = mask.index[row_pos]  # MultiIndex tuples (origin, dest)
    row_lab_arr = row_labels.to_frame(index=False).to_numpy()  # shape (K, 2)

    # rare_idx holds [origin, dest, time_idx_pos]
    rare_idx = np.hstack([row_lab_arr, col_pos.reshape(-1, 1)])

    # 5) Prepare an empty DataFrame (same shape as `sub`) to receive forecasted values.
    rare_trans_prob = sub * np.nan

    # 6) Map row labels to row integer locations in rare_trans_prob.
    row_mi = pd.MultiIndex.from_arrays(
        [rare_idx[:, 0], rare_idx[:, 1]],
        names=['origin', 'dest']
    )
    row_int = rare_trans_prob.index.get_indexer(row_mi)

    # 7) Map column integer positions to actual labels, then to integer locations (safety).
    col_lab = rare_trans_prob.columns[rare_idx[:, 2]]
    col_int = rare_trans_prob.columns.get_indexer(col_lab)

    # 8) Pull forecast values from trans_cube at (origin, dest, time % T).
    T = trans_cube.shape[2]
    origin_idx = rare_idx[:, 0].astype(int)
    dest_idx   = rare_idx[:, 1].astype(int)
    time_idx   = (rare_idx[:, 2].astype(int)) % T
    vals = trans_cube[origin_idx, dest_idx, time_idx]

    # 9) Vectorized assignment into DataFrame values.
    #    Note: we assume all rows/cols exist; if some don't, get_indexer may return -1.
    ok = (row_int >= 0) & (col_int >= 0)
    if not np.all(ok):
        # Keep only valid pairs (present in the DataFrame)
        row_int = row_int[ok]
        col_int = col_int[ok]
        vals    = vals[ok]

    rare_trans_prob.values[row_int, col_int] = vals

    # 10) Return concatenation on columns (aligns by index automatically).
    return pd.concat([fitted_df, rare_trans_prob], axis=0)



def run_oam_pm_pipeline(
    directions_input: Dict[str, pd.DataFrame],
    feature_dict: Dict[str, Dict[str, list]],
    *,
    p_hat_all: Dict[str, Dict[str, pd.DataFrame]],
    group_trans_prob_all: Dict[str, Dict[str, pd.DataFrame]],
    leave_gate_all: Dict[str, Dict[str, pd.DataFrame]],
    end_of_trn: int,
    myrun,
    max_streak: int = 24,
    daily_harmonics: int = 0,
    weekly_harmonics: int = 0,
    deltrend: float = 1,
    delseason: float = 1,
    delregn: float = 1,
    include_intercept: bool = True,
    upweight_pm: bool = False,
    upweight_oam: bool = False,
    upweight_type: int = 1,
    metrics_warmup_steps: int = 288*7
) -> Dict[str, Dict[str, Any]]:
    """
    Build X arrays, fit PM (frozen after training) and OAM (continue from PM) for each type.
    Returns a dict keyed by type_name with PM/OAM results and artifacts.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for type_name, input_transition_ind in directions_input.items():
        # type_name = 'staying_to_other'; input_transition_ind = directions_input[type_name]
        # --- Assemble feature stacks (order matters: kernel/switch, group lags, leave_gate)
        X_array_oam = np.stack(
            [p_hat_all[type_name]['oam']] +
            list(group_trans_prob_all[type_name].values()) +
            [leave_gate_all[type_name]['oam']],
            axis=0
        )
        X_array_pm = np.stack(
            [p_hat_all[type_name]['pm']] +
            list(group_trans_prob_all[type_name].values()) +
            [leave_gate_all[type_name]['pm']],
            axis=0
        )

        # --- Feature names for this type
        feat_cols  = feature_dict[type_name]['feature']
        logit_cols = feature_dict[type_name]['logit_feature']

        # --- Build XY for OAM/PM
        input_df_oam = build_Xy_summary(
            input_transition_ind,
            X_array_oam,
            feat_cols=['kernel_prob','lag1','lag2','leave_gate'],
            logit_cols=['kernel_prob','lag1','lag2'],
            max_streak=max_streak
        )
        input_df_oam['update'] = True

        input_df_pm = build_Xy_summary(
            input_transition_ind,
            X_array_pm,
            feat_cols=['kernel_prob','lag1','lag2','leave_gate'],
            logit_cols=['kernel_prob','lag1','lag2'],
            max_streak=max_streak
        )
        input_df_pm['update'] = False
        input_df_pm.loc[input_df_pm['time_idx'] <= end_of_trn, 'update'] = True

        # --- Fit PM (prior model; features frozen in test)
        mod_pm, forecasts_pm, states_pm = fit_individual_bernoulli_dglm(
            xy=input_df_pm,
            feature_cols=feat_cols,
            daily_harmonics=daily_harmonics,
            weekly_harmonics=weekly_harmonics,
            deltrend=deltrend, delseason=delseason, delregn=delregn,
            include_intercept=include_intercept,
            upweight=upweight_pm, upweight_type=upweight_type
        )
        forecasts_sub_pm = forecasts_pm[forecasts_pm.index >= metrics_warmup_steps]
        results_summary_pm = summarize_binary_predictions(forecasts_sub_pm, states_pm)

        forecasted_prob_df_pm = fill_forecasted_probabilities(
            input_df_pm, forecasts_pm, input_transition_ind,
            od_col="OD_idx", time_col="time_idx", p_hat_col="p_hat"
        )

        # --- OAM: start from PM and continue only on post-train rows
        mod_oam = copy.deepcopy(mod_pm)
        post_mask = (input_df_oam['time_idx'] > end_of_trn)
        mod_oam, forecasts_oam_tail, states_oam = continue_individual_bernoulli_dglm(
            mod_oam,
            xy=input_df_oam.loc[post_mask],
            feature_cols=feat_cols,
            upweight=upweight_oam,
            upweight_type=upweight_type
        )
        # stitch forecasts: PM until end_of_trn, then OAM tail
        forecasts_oam = pd.concat(
            [forecasts_pm[forecasts_pm.index <= end_of_trn], forecasts_oam_tail],
            axis=0
        )
        forecasts_sub_oam = forecasts_oam[forecasts_oam.index >= metrics_warmup_steps]
        results_summary_oam = summarize_binary_predictions(forecasts_sub_oam, states_oam)

        forecasted_prob_df_oam = fill_forecasted_probabilities(
            input_df_oam, forecasts_oam, input_transition_ind,
            od_col="OD_idx", time_col="time_idx", p_hat_col="p_hat"
        )
        
        forecast_results_summary = pd.concat([results_summary_pm.T, results_summary_oam.T], axis = 1)
        forecast_results_summary.columns = ['pm', 'oam']
        
        # --- Store in results
        results[type_name] = {
            # models and forecasts
            "mod_pm": mod_pm,
            "forecasts_pm": forecasts_pm,
            "states_pm": states_pm,
            "mod_oam": mod_oam,
            "forecasts_oam": forecasts_oam,
            "states_oam": states_oam,
            # summaries
            "results_summary": forecast_results_summary,
            # filled probabilities
            "forecasted_prob_df_pm": forecasted_prob_df_pm,
            "forecasted_prob_df_oam": forecasted_prob_df_oam,
            # inputs (optional, handy for debugging)
            "input_df_pm": input_df_pm,
            "input_df_oam": input_df_oam,
        }

    return results




def log_cumulative_bayes_factor(log_H):
    """
    Compute log L_t sequentially using the DP recursion from Theorem 11.3.

    Parameters
    ----------
    log_H : array-like
        log of individual Bayes factors h_t = log H_t

    Returns
    -------
    log_L : np.ndarray
        log of cumulative Bayes factors L_t at each t
    run_length : np.ndarray
        run-length l_t (number of consecutive obs in the current run)
    """
    n = len(log_H)
    log_L = np.zeros(n)
    run_length = np.zeros(n, dtype=int)

    log_L[0] = log_H[0]
    run_length[0] = 1

    for t in range(1, n):
        log_L[t] = log_H[t] + min(0, log_L[t-1])
        run_length[t] = run_length[t-1] + 1 if log_L[t-1] < 0 else 1

    return log_L, run_length