import numpy as np
import pandas as pd
from typing import Union
# ------------------------------------------------------------
# Utilities for individual-level modeling
# ------------------------------------------------------------

def _run_length_series(y_row: pd.Series, identifier: Union[int, float, bool] = 1) -> pd.Series:
    """
    Run-length of consecutive `identifier` values.

    Behavior
    --------
    Let `id` = identifier (e.g., 1 for 'stay', 0 for 'stay' if you invert).

    - If y_t == id:
        - If align_to_transition=False: r_t = current run length including this step (1,2,3,...)
        - If align_to_transition=True : r_t = fill_non_transition (we only write at the transition)
    - If y_t != id and not NaN (i.e., transition off the run):
        - If align_to_transition=False: r_t = previous run length (the count just before this step)
        - If align_to_transition=True : r_t = previous run length (written at the transition time)
      Then the counter resets.
    - If y_t is NaN: r_t = fill_non_transition and the counter resets.

    Notes
    -----
    - Output dtype is float so we can represent NaNs.
    - Use `identifier=0` if you want run lengths of consecutive zeros.

    Parameters
    ----------
    y : pd.Series of scalars (e.g., {0,1,NaN})
    identifier : value to count consecutive runs for (default: 1)
    align_to_transition : if True, only write the run length at the *first non-identifier* step.
    fill_non_transition : value to write at non-transition times when align_to_transition=True
                          (default NaN; set to 0.0 if you prefer zeros)

    Returns
    -------
    pd.Series of floats, aligned to y.index.
    """
    r = np.empty(len(y_row), dtype='float')
    cnt = 0
    for i, v in enumerate(y_row.values):
        if pd.isna(v):
            cnt = 0
            r[i] = np.nan
        elif v == identifier:
            cnt += 1
            r[i] = cnt
        else:  # v == 0
            r[i] = cnt
            cnt = 0
    return pd.Series(r, index=y_row.index)

def _most_recent_average_mean_stay(
    y: pd.Series,
    decay: float = 0.8,
    count_zero_length: bool = False,
    identifier: Union[int, float, bool] = 1
) -> pd.Series:
    """
    Exponentially decayed mean of completed 'stay' lengths, updated online.

    A 'stay' is a consecutive run of `identifier` values. The mean updates only
    when a run finishes (i.e., the first non-identifier after the run).
    NaNs reset the current run and do not update the mean.

    Parameters
    ----------
    y : pd.Series
        Time-ordered series with values in {0,1,NaN} (or comparable scalars).
    decay : float, default 0.8
        Exponential decay applied once per completed run. Recent runs weigh more.
        decay=1.0 reduces to the ordinary expanding mean of completed runs.
    count_zero_length : bool, default False
        If True, a run of length 0 (i.e., a non-identifier immediately) contributes 0.
    identifier : {0,1,...}, default 1
        Value that defines the 'stay' run (use 0 if your stays are zeros).

    Returns
    -------
    pd.Series
        At each time step i, the decayed mean of completed run lengths observed
        up to and including i. NaN until the first completed run is observed.
    """
    y_arr = y.to_numpy()

    current_len = 0.0          # length of the ongoing run of `identifier`
    w_sum = 0.0                # decayed sum of run weights
    wL_sum = 0.0               # decayed sum of (weight * run_length)
    mean_out = pd.Series(np.nan, index=y.index, dtype=float)

    for i, v in enumerate(y_arr):
        if pd.isna(v):
            # break: reset ongoing run; no mean update
            current_len = 0.0

        elif v == identifier:
            # extend the ongoing run
            current_len += 1.0

        else:
            # run ends here -> update decayed mean with the completed length
            if current_len > 0.0 or count_zero_length:
                w_sum  = decay * w_sum  + 1.0
                wL_sum = decay * wL_sum + float(current_len)
            current_len = 0.0

        mean_out.iat[i] = (wL_sum / w_sum) if w_sum > 0.0 else np.nan

    return mean_out

def kernel_smoothed_prob_binary(
    y: pd.Series,
    freq: str = "5min",
    h_minutes: int = 30,
    window_minutes: int = 60,
    lambda_days: float = 5.0,
    include_same_day: bool = True,
    exclude_current_slot: bool = True,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> pd.Series:
    """
    Kernel-smoothed Bernoulli probability around each time point.
    y is assumed to be 0/1 with a DatetimeIndex.

    Parameters
    ----------
    y : pd.Series
        Binary time series with DatetimeIndex, e.g. 0/1 indicating presence in a zone.
    freq : str
        Time resolution, e.g. '5min'.
    h_minutes : int
        Gaussian kernel bandwidth in minutes (time-of-day direction).
    window_minutes : int
        Half window width around each time in minutes.
    lambda_days : float
        Exponential decay constant (in days) for past-day weights.
    include_same_day : bool
        Whether to include same-day past slots in the kernel.
    exclude_current_slot : bool
        If True, the current slot is excluded from the kernel weights (avoids leakage).
    alpha0, beta0 : float
        Beta prior hyperparameters for smoothing.

    Returns
    -------
    pd.Series
        Smoothed probabilities aligned to y.index.
    """
    # Ensure y is clean
    y = y.sort_index().dropna()
    T = len(y)
    if T == 0:
        return pd.Series([], index=y.index, dtype=float)

    # Convert to day-slot grid
    slot_td = pd.Timedelta(freq)
    slot_minutes = int(slot_td.total_seconds() // 60)
    days = pd.DatetimeIndex(sorted(pd.unique(y.index.normalize())))
    D = len(days)
    S = int(24 * 60 // slot_minutes)

    # Index maps
    day_to_i = {d: i for i, d in enumerate(days)}
    day_idx = np.array([day_to_i[t.normalize()] for t in y.index], dtype=int)
    minutes_since_midnight = ((y.index - y.index.normalize()).total_seconds() // 60).astype(int)
    slot_idx = (minutes_since_midnight // slot_minutes).astype(int)

    # Fill matrix with 0/1 and valid mask
    Z = np.full((D, S), -1, dtype=np.int8)
    valid = np.zeros((D, S), dtype=bool)
    Z[day_idx, slot_idx] = y.values.astype(np.int8)
    valid[day_idx, slot_idx] = True

    # Convert kernel parameters to slot units
    h_bins = max(1, int(round(h_minutes / slot_minutes)))
    W_bins = max(1, int(round(window_minutes / slot_minutes)))

    # Prepare output
    P = np.full((D, S), np.nan, dtype=float)
    day_idx_arr = np.arange(D)
    slot_idx_arr = np.arange(S)

    for d in range(D):
        day_lag = d - day_idx_arr
        w_row = np.exp(-day_lag / lambda_days)
        w_row[day_lag < 0] = 0.0
        if not include_same_day:
            w_row[day_lag == 0] = 0.0
        w_row = w_row[:, None]

        for s in range(S):
            if not valid[d, s]:
                continue

            delta = np.abs(slot_idx_arr - s)
            delta = np.minimum(delta, S - delta)
            w_col = np.exp(- (delta / h_bins) ** 2)
            w_col[delta > W_bins] = 0.0
            w_col = w_col[None, :]

            W = (w_row * w_col) * valid
            if include_same_day:
                if exclude_current_slot:
                    W[d, s:] = 0.0
                else:
                    W[d, s+1:] = 0.0

            if not np.any(W):
                P[d, s] = alpha0 / (alpha0 + beta0)
                continue

            alpha = alpha0 + (W * (Z == 1)).sum()
            beta = beta0 + (W * (Z == 0)).sum()
            P[d, s] = alpha / (alpha + beta)

    p_hat_vals = P[day_idx, slot_idx]
    return pd.Series(p_hat_vals, index=y.index)


def leave_pressure(
    y: pd.Series,
    decay: float = 0.8,
    threshold: float = 6.0,
    count_zero_length: bool = False,
    identifier: Union[int, float, bool] = 1
) -> pd.Series:
    """
    Compute 'leave pressure' feature:
    - Uses exponentially decayed mean of past stay lengths (1-run lengths),
    - Measures how much current stay length exceeds (mean - threshold),
    - Accumulates only after the threshold is crossed,
    - Shifts by 1 to align with transition events.

    Parameters
    ----------
    y : pd.Series
        0/1/NaN time series (1 = staying, 0 = leaving, NaN = break).
    decay : float, default 0.8
        Exponential decay factor for weighting past stay lengths.
    threshold : float, default 6.0
        How far below the mean to start accumulating pressure.
    count_zero_length : bool, default False
        If True, zeros not preceded by any 1 count as a run of length 0.

    Returns
    -------
    pd.Series
        leave_pressure values aligned to t-1 (for modeling hazard at t).
    """
    
    mean_out = _most_recent_average_mean_stay(y, decay, count_zero_length, identifier)
    # --- compute stay length ---
    stay_len = _run_length_series(y, identifier)

    # --- leave pressure feature ---
    pressure = np.maximum(stay_len - (mean_out - threshold), 0)

    # shift to align with transition events
    return pressure



def kstep_trans_feature(
    lag: int,
    agent_trajectory: pd.Series,
    transition_ind_matrix: pd.DataFrame,
    cube,
    min_zone_id: int = -121
) -> pd.DataFrame:
    """
    Retrieve the lagged k-step (k >= 2) transition probability features from 
    transition probability array and align them to each direction.

    Parameters
    ----------
    lag : int
        Lag (in time steps) between origin and destination used to compute
        k-step transition probabilities.
    agent_trajectory : pd.Series
        Agent's zone trajectory over time. Each value is a zone ID.
    transition_ind_matrix : pd.DataFrame
        K×T indicator matrix for transitions of interest (e.g., staying_to_other),
        where rows are OD pairs and columns are time steps.
    cube : np.ndarray
        K_step_trans_mat[lag] is of shape (N, N, T),
        giving the probability of transitioning from zone i to j at time t.
    min_zone_id : int, default 0
        Minimum zone ID offset (used to shift zone IDs to zero-based indexing).

    Returns
    -------
    pd.DataFrame
        K×T DataFrame of transition features, aligned with
        `transition_ind_matrix`.
    """
    # lag = 2
    # agent_trajectory = agent_i_full
    # transition_ind_matrix = transition_counts_cascade
    # cube = K_step_trans_mat[lag][:,:,:len(agent_i_full)-1]
    OD_coord = pd.DataFrame(
        pd.concat([agent_trajectory.shift(lag), agent_trajectory], axis=1).values[lag:, :].astype(int) - min_zone_id,
        columns=['origin', 'dest']
    )
    # cube = K_step_trans_mat[lag]
    prob = cube[
        OD_coord['origin'].values,
        OD_coord['dest'].values,
        np.arange(cube.shape[2])
    ]
    prob = np.concatenate([np.zeros(lag - 1,), prob])
    return (transition_ind_matrix.notna()).astype(int).mul(prob, axis=1)

'''
Utiltiy functions to retrieve the group-level transition probabilities
'''

def build_full_transition(fcast_multinom_df: pd.DataFrame, eps: float = 1e-12, tol: float = 1e-8):
    """
    Expand partial (origin,dest)×time probs into a full N×N×T tensor without looping over origins.
    For each time t, fill an N×N matrix; for each row, distribute the leftover probability
    equally among zero-prob destinations so rows sum to 1.

    Assumes:
    - Index names are ['origin','dest'].
    - N = number of unique origins (level 0).
    """
    if list(fcast_multinom_df.index.names) != ['origin', 'dest']:
        raise ValueError("Index must be a MultiIndex with names ['origin','dest'].")

    # Zone order and dimensions (per your note: N from level(0))
    zones = fcast_multinom_df.index.get_level_values('origin').unique().sort_values()
    N = len(zones)
    times = fcast_multinom_df.columns
    T = len(times)

    P = np.empty((N, N, T), dtype=float)

    for ti, t in enumerate(times):
        # N×N matrix for this time: rows=origin, cols=dest (missing -> 0)
        # t = times[0]
        M = (
            fcast_multinom_df[t]
            .unstack('dest')
            .reindex(index=zones, columns=zones)
        ).fillna(0)
        A = M.to_numpy(copy=True)  # shape (N, N)

        # Leftover per row and zero slots
        row_sum = A.sum(axis=1, keepdims=True)
        zero_mask = (A <= eps)
        n_zero = zero_mask.sum(axis=1, keepdims=True)
        remainder = 1.0 - row_sum

        # Distribute leftover only where there are zeros (negative remainder is clipped to 0 here)
        add = np.divide(np.maximum(remainder, 0.0), n_zero,
                        out=np.zeros_like(remainder), where=n_zero > 0)
        A = A + add * zero_mask

        if not np.allclose(A.sum(axis=1), 1.0, atol=tol):
            bad_rows = np.where(~np.isclose(A.sum(axis=1), 1.0, atol=tol))[0]
            raise ValueError(
                f"Row sums at time {t} not equal to 1 within tolerance "
                f"(bad rows: {bad_rows}, values: {A.sum(axis=1)[bad_rows]})"
            )
            
        P[:, :, ti] = A

    return {
        'transition_matrix': P, 
            'zone_idx': zones
            }

def long_matrix_array_multipler(A: np.ndarray, B: np.ndarray, batch_size: int =1000)-> np.ndarray:
    """
    Perform batched matrix multiplication along the third axis of A and B
    using chunked processing to reduce overhead and improve performance.

    Computes:
        C[:, :, i] = A[:, :, i] @ B[:, :, i]
    for i = 0 ... T-1, where T = A.shape[2].
    
    It takes about 2.5min to mutiply two arrays with size 215*215*8063
    Parameters
    ----------
    A : np.ndarray of shape (N, N, T)
        Left operand batch of matrices.
    B : np.ndarray of shape (N, N, T)
        Right operand batch of matrices.
    batch_size : int, default=1000
        Number of slices to process in one batch (tune for speed).

    Returns
    -------
    C : np.ndarray of shape (N, N, T)
        Result of batched matrix multiplication A @ B.
    """
    N, _, T = A.shape
    C = np.empty_like(A)          # preallocate result array
    # Process the T slices in chunks to reduce per-call overhead
    for start in range(0, T, batch_size):
        end = min(T, start + batch_size)
        # einsum computes batch matmul for the chunk [start:end]
        C[:, :, start:end] = np.einsum('ijt,jkt->ikt',
                                       A[:, :, start:end],
                                       B[:, :, start:end])

    return C


def multistep_transition(P: np.ndarray, K: int, tol: float = 1e-9):
    """
    Time-varying multi-step transitions for P with shape (N, N, T), time on the last axis.
    Returns a dict {k: M_k} where M_k has shape (N, N, T-k+1)
    and M_k[:, :, t] = P[:, :, t] @ P[:, :, t+1] @ ... @ P[:, :, t+k-1].

    Strictly validates that each row sums to 1 (within `tol`). No renormalization.
    """
    if K < 2:
        raise ValueError("K must be >= 2.")
    N, N2, T = P.shape
    if N != N2:
        raise ValueError("P must be N×N×T.")
    if K > T:
        raise ValueError("K must be <= T.")

    out = {}

    # k = 2: for t = 0..T-2
    # A = P[:,:,0:T-1], B = P[:,:,1:T]
    # einsum 'ijt,jkt->ikt' does: out[i,k,t] = sum_j A[i,j,t] * B[j,k,t]
    # M_prev = np.einsum('ijt,jkt->ikt', P[:, :, :-1], P[:, :, 1:])  # (N, N, T-1)
    M_prev = long_matrix_array_multipler(P[:, :, :-1], P[:, :, 1:])  # (N, N, T-1)

    # row-sum check
    if not np.allclose(M_prev.sum(axis=1), 1.0, atol=tol):
        bad_mask = ~np.isclose(M_prev.sum(axis=1), 1.0, atol=tol)
        bad_rows, bad_ts = np.where(bad_mask)
        raise ValueError(
            f"Row sums deviate from 1 for k=2. "
            f"Examples (row, t): {list(zip(bad_rows[:10], bad_ts[:10]))}"
        )
    out[2] = M_prev

    # k = 3..K, DP: M_k[:, :, t] = M_{k-1}[:, :, t] @ P[:, :, t+k-1]
    for k in range(3, K + 1):
        # Align time: use M_prev[:, :, :-1] with P[:, :, k-1:]
        M_k = long_matrix_array_multipler(M_prev[:, :, :-1], P[:, :, k-1:])  # (N,N,T-k+1)

        if not np.allclose(M_k.sum(axis=1), 1.0, atol=tol):
            bad_mask = ~np.isclose(M_k.sum(axis=1), 1.0, atol=tol)
            bad_rows, bad_ts = np.where(bad_mask)
            raise ValueError(
                f"Row sums deviate from 1 for k={k}. "
                f"Examples (row, t): {list(zip(bad_rows[:10], bad_ts[:10]))}"
            )

        out[k] = M_k
        M_prev = M_k  # DP carry

    return out




def signed_leave_gate(df: pd.DataFrame, decay: float = 0.8, threshold: float = 6.0) -> pd.DataFrame:
    """
    For each (origin, dest) row in df (0/1/NaN), compute the indicator:
        I_t = 1{ len_cur_stay_t > cur_stay_avg_t - threshold }
    where len_cur_stay and cur_stay_avg use:
        - identifier = 1 if origin == dest
        - identifier = 0 otherwise

    Finally, return:
        -I_t if origin == dest,
        +I_t if origin != dest.

    Parameters
    ----------
    df : pd.DataFrame
        K×T matrix with rows indexed by (origin, dest).
    decay : float
        Exponential decay for the rolling mean of stay lengths.
    threshold : float
        Margin subtracted from the decayed mean.

    Returns
    -------
    pd.DataFrame
        K×T DataFrame with signed indicators per row.
    """
    def _row_to_signed_indicator(row: pd.Series) -> pd.Series:
        o, d = row.name
        identifier = 1 if o == d else 0  

        L = _run_length_series(row, identifier=identifier)
        M = _most_recent_average_mean_stay(row, decay=decay, identifier=identifier)

        diff = L - (M - threshold)
        ind = (diff > 0).astype(float)
        ind[np.isnan(diff)] = np.nan  # preserve NaNs

        sign = -1.0 if o == d else 1.0
        return ind * sign

    return df.apply(_row_to_signed_indicator, axis=1)


# def switch_prob(
#     y: pd.Series,
#     freq: str = "5min",
#     window_minutes: int = 60,
# ) -> pd.Series:
#     """
#     For each observed timestamp t with day i and slot j, compute:

#       proportion(i, j) =
#         (# of days r in [0..i] that contain at least one '1' within circular window [j-eps, j+eps])
#         /
#         (# of days r in [0..i] whose window [j-eps, j+eps] has at least one observed slot (not all NaN))

#     Circular window across time-of-day. Days whose window is all-NaN are excluded from the denominator.
#     Missing slots are allowed; the last day can be partially filled.

#     Parameters
#     ----------
#     y : pd.Series
#         Binary series (0/1/NaN) with a DatetimeIndex on a regular grid given by `freq`.
#         NaNs mean "no observation".
#     freq : str
#         Grid frequency.
#     window_minutes : int
#         Half-window (±) in minutes around slot j (circular).

#     Returns
#     -------
#     pd.Series
#         Proportion values aligned to y.index (NaN when no usable days yet).
#     """
#     # keep NaNs (we need to know what's unobserved), but ensure sorted
#     y = y.sort_index()
#     if len(y) == 0:
#         return pd.Series([], index=y.index, dtype=float)

#     # Grid specification
#     slot_td = pd.Timedelta(freq)
#     slot_minutes = int(slot_td.total_seconds() // 60)
#     S = int(round(24 * 60 / slot_minutes))          # slots/day
#     eps_slots = max(0, int(round(window_minutes / slot_minutes)))

#     # Day/slot mapping
#     days = pd.DatetimeIndex(sorted(pd.unique(y.index.normalize())))
#     D = len(days)
#     day_to_i = {d: i for i, d in enumerate(days)}
#     day_idx = np.array([day_to_i[t.normalize()] for t in y.index], dtype=int)
#     minutes_since_midnight = ((y.index - y.index.normalize()).total_seconds() // 60).astype(int)
#     slot_idx = (minutes_since_midnight // slot_minutes).astype(int)

#     # Build D×S matrices:
#     #   V: observed mask (True where y is 0 or 1)
#     #   B: event mask (1 where y==1, else 0)
#     V = np.zeros((D, S), dtype=bool)
#     B = np.zeros((D, S), dtype=np.uint8)

#     vals = y.to_numpy()
#     obs_mask = ~pd.isna(vals)
#     ones_mask = (vals == 1)

#     V[day_idx[obs_mask], slot_idx[obs_mask]] = True
#     B[day_idx[ones_mask], slot_idx[ones_mask]] = 1

#     # For each column j, compute across days:
#     #   has_one[j,i] = any 1 in window on day i
#     #   used_day[j,i] = window has at least one observed slot on day i (not all NaN)
#     rows_have_one = np.zeros((S, D), dtype=np.uint8)
#     rows_used     = np.zeros((S, D), dtype=np.uint8)

#     base = np.arange(-eps_slots, eps_slots + 1)  # circular window offsets

#     for j in range(S):
#         cols = (j + base) % S
#         # any() across window columns per day
#         has_one = B[:, cols].any(axis=1)           # bool (D,)
#         has_obs = V[:, cols].any(axis=1)           # bool (D,)

#         rows_have_one[j, :] = has_one.astype(np.uint8)
#         rows_used[j, :]     = has_obs.astype(np.uint8)

#     # Cumulative along days: up to and including day i
#     cum_have_one = np.cumsum(rows_have_one, axis=1)  # (S, D)
#     cum_used     = np.cumsum(rows_used,     axis=1)  # (S, D)

#     num = cum_have_one[slot_idx, day_idx].astype(float)
#     den = cum_used[slot_idx, day_idx].astype(float)

#     # proportion = num / den; if den==0 -> NaN
#     proportions = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=(den > 0))

#     return pd.Series(proportions, index=y.index, dtype=float)

# def switch_prob(
#     y: pd.Series,
#     freq: str = "5min",
#     window_minutes: int = 60,
#     delta: float = 1.0,
# ) -> pd.Series:
#     """
#     For each observed timestamp t with day i and slot j, compute an exponentially
#     discounted historical proportion over days:

#       proportion(i, j) =
#         (sum_{d=0..i} [ any(1 in window on day d) * delta^(i-d) ])
#         /
#         (sum_{d=0..i} [ any(obs in window on day d) * delta^(i-d) ])

#     - Circular window across time-of-day (±window_minutes).
#     - Days whose window is all-NaN are excluded via the denominator path.
#     - Missing slots are allowed; the last day can be partially filled.
#     - delta=1.0 reproduces the original cumulative definition without decay.

#     Parameters
#     ----------
#     y : pd.Series
#         Binary series (0/1/NaN) with a DatetimeIndex on a regular grid given by `freq`.
#         NaNs mean "no observation".
#     freq : str
#         Grid frequency (e.g., "5min").
#     window_minutes : int
#         Half-window (±) in minutes around slot j (circular).
#     delta : float, default 1.0
#         Exponential decay per day back in time. Typical 0 < delta ≤ 1.
#         delta=1.0 => no decay; smaller delta downweights older days.

#     Returns
#     -------
#     pd.Series
#         Proportion values aligned to y.index (NaN when no usable days yet).
#     """
#     # keep NaNs (we need to know what's unobserved), but ensure sorted
#     y = y.sort_index()
#     if len(y) == 0:
#         return pd.Series([], index=y.index, dtype=float)

#     # Grid specification
#     slot_td = pd.Timedelta(freq)
#     slot_minutes = int(slot_td.total_seconds() // 60)
#     if slot_minutes <= 0:
#         raise ValueError("freq must be a positive time interval.")
#     S = int(round(24 * 60 / slot_minutes))          # slots/day
#     eps_slots = max(0, int(round(window_minutes / slot_minutes)))

#     # Day/slot mapping
#     days = pd.DatetimeIndex(sorted(pd.unique(y.index.normalize())))
#     D = len(days)
#     day_to_i = {d: i for i, d in enumerate(days)}
#     day_idx = np.array([day_to_i[t.normalize()] for t in y.index], dtype=int)
#     minutes_since_midnight = ((y.index - y.index.normalize()).total_seconds() // 60).astype(int)
#     slot_idx = (minutes_since_midnight // slot_minutes).astype(int)

#     # Build D×S matrices:
#     #   V: observed mask (True where y is 0 or 1)
#     #   B: event mask (1 where y==1, else 0)
#     V = np.zeros((D, S), dtype=bool)
#     B = np.zeros((D, S), dtype=np.uint8)

#     vals = y.to_numpy()
#     obs_mask = ~pd.isna(vals)
#     ones_mask = (vals == 1)

#     V[day_idx[obs_mask], slot_idx[obs_mask]] = True
#     B[day_idx[ones_mask], slot_idx[ones_mask]] = 1

#     # For each column j, compute across days:
#     #   has_one[j,i] = any 1 in window on day i
#     #   used_day[j,i] = window has at least one observed slot on day i (not all NaN)
#     rows_have_one = np.zeros((S, D), dtype=np.uint8)
#     rows_used     = np.zeros((S, D), dtype=np.uint8)

#     base = np.arange(-eps_slots, eps_slots + 1)  # circular window offsets
#     for j in range(S):
#         cols = (j + base) % S
#         has_one = B[:, cols].any(axis=1)   # bool (D,)
#         has_obs = V[:, cols].any(axis=1)   # bool (D,)
#         rows_have_one[j, :] = has_one.astype(np.uint8)
#         rows_used[j, :]     = has_obs.astype(np.uint8)

#     # Exponentially discounted cumulative sums along days:
#     # out[:, i] = rows[:, i] + delta * out[:, i-1]
#     # This yields sum_{d<=i} rows[:, d] * delta^(i-d)
#     S_, D_ = rows_have_one.shape
#     num_w = np.zeros((S_, D_), dtype=float)
#     den_w = np.zeros((S_, D_), dtype=float)

#     if D_ > 0:
#         num_w[:, 0] = rows_have_one[:, 0].astype(float)
#         den_w[:, 0] = rows_used[:, 0].astype(float)
#         if D_ > 1:
#             for i in range(1, D_):
#                 num_w[:, i] = rows_have_one[:, i].astype(float) + delta * num_w[:, i-1]
#                 den_w[:, i] = rows_used[:, i].astype(float)     + delta * den_w[:, i-1]

#     num = num_w[slot_idx, day_idx]
#     den = den_w[slot_idx, day_idx]

#     proportions = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=(den > 0))
#     return pd.Series(proportions, index=y.index, dtype=float)


def switch_prob(
    y: pd.Series,
    freq: str = "5min",
    window_minutes: int = 60,
    delta: float = 1,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> pd.Series:
    """
    For each timestamp t with day i and slot j, compute an exponentially
    discounted historical proportion using ONLY days < i (i.e., up to yesterday):

      hist_num(i, j) =
        sum_{d=0..i-1} [ any(1 in window on day d, slot j) * delta^(i-d) ]

      hist_den(i, j) =
        sum_{d=0..i-1} [ any(obs in window on day d, slot j) * delta^(i-d) ]

    Beta-Bernoulli smoothing (alpha, beta) is applied on top of the discounted counts:
      proportion(i, j) = (alpha + hist_num) / (alpha + beta + hist_den)

    Notes
    -----
    - "window" is circular across time-of-day: ±window_minutes around slot j.
    - Days whose window is all-NaN are excluded via the denominator path.
    - Missing slots are allowed; the last day can be partially filled.
    - delta in (0,1] downweights older days; delta=1.0 gives no decay.
    - Because of Beta smoothing, if there is no usable history, the output is
      alpha / (alpha + beta), default 0.5.

    Parameters
    ----------
    y : pd.Series
        Binary series (0/1/NaN) with a DatetimeIndex on a regular grid given by `freq`.
        NaNs mean "no observation".
    freq : str
        Grid frequency (e.g., "5min").
    window_minutes : int
        Half-window (±) in minutes around slot j (circular).
    delta : float, default 1.0
        Exponential decay per day back in time. Typical 0 < delta ≤ 1.
    alpha, beta : float, default 0.5
        Beta-Bernoulli smoothing hyperparameters.

    Returns
    -------
    pd.Series
        Smoothed proportion values aligned to y.index (always finite due to smoothing).
    """
    # keep NaNs (we need to know what's unobserved), but ensure sorted
    y = y.sort_index()
    if len(y) == 0:
        return pd.Series([], index=y.index, dtype=float)

    # Grid specification
    slot_td = pd.Timedelta(freq)
    slot_minutes = int(slot_td.total_seconds() // 60)
    if slot_minutes <= 0:
        raise ValueError("freq must be a positive time interval.")
    S = int(round(24 * 60 / slot_minutes))          # slots/day
    eps_slots = max(0, int(round(window_minutes / slot_minutes)))

    # Day/slot mapping
    days = pd.DatetimeIndex(sorted(pd.unique(y.index.normalize())))
    D = len(days)
    day_to_i = {d: i for i, d in enumerate(days)}
    day_idx = np.array([day_to_i[t.normalize()] for t in y.index], dtype=int)
    minutes_since_midnight = ((y.index - y.index.normalize()).total_seconds() // 60).astype(int)
    slot_idx = (minutes_since_midnight // slot_minutes).astype(int)

    # Build D×S matrices:
    #   V: observed mask (True where y is 0 or 1)
    #   B: event mask (1 where y==1, else 0)
    V = np.zeros((D, S), dtype=bool)
    B = np.zeros((D, S), dtype=np.uint8)

    vals = y.to_numpy()
    obs_mask = ~pd.isna(vals)
    ones_mask = (vals == 1)

    V[day_idx[obs_mask], slot_idx[obs_mask]] = True
    B[day_idx[ones_mask], slot_idx[ones_mask]] = 1

    # For each slot j, compute (across days):
    #   rows_have_one[j, i] = any 1 in circular window on day i
    #   rows_used[j, i]     = window has at least one observed slot on day i
    rows_have_one = np.zeros((S, D), dtype=np.uint8)
    rows_used     = np.zeros((S, D), dtype=np.uint8)

    base = np.arange(-eps_slots, eps_slots + 1)  # circular window offsets
    for j in range(S):
        cols = (j + base) % S
        has_one = B[:, cols].any(axis=1)   # bool (D,)
        has_obs = V[:, cols].any(axis=1)   # bool (D,)
        rows_have_one[j, :] = has_one.astype(np.uint8)
        rows_used[j, :]     = has_obs.astype(np.uint8)

    # Exponentially discounted cumulative sums along days (forward recursion):
    # cum[:, i] = rows[:, i] + delta * cum[:, i-1]
    # => cum[:, i] = sum_{d<=i} rows[:, d] * delta^(i-d)
    S_, D_ = rows_have_one.shape
    num_w = np.zeros((S_, D_), dtype=float)
    den_w = np.zeros((S_, D_), dtype=float)

    if D_ > 0:
        num_w[:, 0] = rows_have_one[:, 0].astype(float)
        den_w[:, 0] = rows_used[:, 0].astype(float)
        for i in range(1, D_):
            num_w[:, i] = rows_have_one[:, i].astype(float) + delta * num_w[:, i-1]
            den_w[:, i] = rows_used[:, i].astype(float)     + delta * den_w[:, i-1]

    # For timestamp at (day=i, slot=j), we want ONLY days < i.
    # Using the identity:
    #   sum_{d<=i-1} rows[d] * delta^(i-d) = delta * (sum_{d<=i-1} rows[d] * delta^{(i-1)-d})
    # = delta * cum[:, i-1].
    prev_day_idx = day_idx - 1

    hist_num = np.zeros_like(prev_day_idx, dtype=float)
    hist_den = np.zeros_like(prev_day_idx, dtype=float)

    valid_prev = prev_day_idx >= 0
    if valid_prev.any():
        hist_num[valid_prev] = delta * num_w[slot_idx[valid_prev], prev_day_idx[valid_prev]]
        hist_den[valid_prev] = delta * den_w[slot_idx[valid_prev], prev_day_idx[valid_prev]]

    # Beta-Bernoulli smoothing
    smoothed_num = alpha + hist_num
    smoothed_den = alpha + beta + hist_den

    proportions = smoothed_num / smoothed_den
    return pd.Series(proportions, index=y.index, dtype=float)

def freeze_training_features(
    df: pd.DataFrame,
    daily_len: int,
    n_days: int,
    end_idx: int = None
) -> pd.DataFrame:
    """
    Duplicate the last `daily_len` columns of a DataFrame for `n_days` times.
    Useful for creating a testing-period feature panel by repeating the last day.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with time-indexed columns (e.g., features over time).
    daily_len : int
        Number of columns representing one day.
    n_days : int
        Number of future days to duplicate.
    end_idx : int, optional
        End index (exclusive) for the training period. Defaults to the end of df.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (df.shape[0], daily_len * n_days) with duplicated columns.
        The index is copied from `df.index`.
        The columns are taken from the original future period (or generated if missing).
    """
    if end_idx is None:
        end_idx = df.shape[1]

    start_idx = end_idx - daily_len
    base_block = df.values[:, start_idx:end_idx]

    # Tile horizontally to duplicate for n_days
    tiled_vals = np.tile(base_block, (1, n_days))

    # Build new column labels
    new_cols = df.columns[end_idx:end_idx + daily_len * n_days]

    return pd.DataFrame(
        tiled_vals,
        index=df.index,
        columns=new_cols
    )
  
def compute_leave_gate_all(
    directions_by_type: dict[str, pd.DataFrame],
    daily_len: int,
    test_days: int,
    end_of_trn: int,
    decay: float = 0.8,
    threshold: float = 18.0
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Compute leave-gate features (OAM and PM) for each transition type
    in `directions_by_type`.

    Returns a dictionary:
      {
        'staying_to_self': {'oam': ..., 'pm': ...},
        'staying_to_other': {'oam': ..., 'pm': ...},
        'passing_to_any': {'oam': ..., 'pm': ...},
        ...
      }
    """
    results = {}
    for key, input_transition_ind in directions_by_type.items():
        # compute leave gate for this key
        leave_gate_oam = signed_leave_gate(
            input_transition_ind.fillna(0),
            decay=decay,
            threshold=threshold
        ).fillna(0)

        leave_gate_pm_frozen = freeze_training_features(
            leave_gate_oam,
            daily_len=daily_len,
            n_days=test_days,
            end_idx=end_of_trn
        )
        leave_gate_pm = pd.concat(
            [leave_gate_oam.iloc[:, :end_of_trn], leave_gate_pm_frozen],
            axis=1
        )

        results[key] = {
            "oam": leave_gate_oam,
            "pm": leave_gate_pm
        }

    return results

def compute_empirical_transition_prob_all(
    directions_by_type: dict[str, pd.DataFrame],
    daily_len: int,
    test_days: int,
    end_of_trn: int,
    # staying_to_self params
    freq: str = "5min",
    h_minutes: int = 15,
    window_minutes_self: int = 90,
    lambda_days: float = 14.0,
    include_same_day: bool = False,
    alpha0: float = 0.5,
    beta0: float = 0.5,
    # other types params
    window_minutes_other: int = 60,
    ewm_span: int = 12
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Compute empirical transition probabilities (OAM and PM) for each transition type.
    """
    results = {}

    for key, input_transition_ind in directions_by_type.items():
        if key == "staying_to_self":
            p_hat_oam = input_transition_ind.apply(
                lambda row: kernel_smoothed_prob_binary(
                    row.fillna(0),
                    freq=freq,
                    h_minutes=h_minutes,
                    window_minutes=window_minutes_self,
                    lambda_days=lambda_days,
                    include_same_day=include_same_day,
                    alpha0=alpha0,
                    beta0=beta0
                ) * (row.notna()).astype(int),
                axis=1
            )
        else:
            p_hat_oam = input_transition_ind.apply(
                lambda row: switch_prob(
                    row, window_minutes=window_minutes_other),
                axis=1
            )
            p_hat_oam = p_hat_oam.T.ewm(span=ewm_span).mean().T

        p_hat_pm_frozen = freeze_training_features(
            p_hat_oam,
            daily_len=daily_len,
            n_days=test_days,
            end_idx=end_of_trn
        )
        p_hat_pm = pd.concat(
            [p_hat_oam.iloc[:, :end_of_trn], p_hat_pm_frozen],
            axis=1
        )

        results[key] = {
            "oam": p_hat_oam,
            "pm": p_hat_pm
        }

    return results
