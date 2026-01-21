'''
This file contains the functions to process agent locations into occupancy and transition counts
'''
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import scipy
from functools import reduce
def timewise_transition_counts_sparse_streaming(df, lag=1, include_self=True):
    """
    Memory-safe, sparse-only transition counter.
    Rows = observed (origin_zone, dest_zone) pairs (sparse),
    Cols = destination time slices (df.columns[lag:]).
    Never builds dense K*K or dense (agents × time) intermediates.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = agents, columns = time slices (sorted), values = zone ids (ints).
    lag : int
        Transition lag: counts transitions from column t -> t+lag.
    include_self : bool
        Keep or drop self transitions (origin == dest).

    Returns
    -------
    M : scipy.sparse.csr_matrix
        Shape = (n_pairs_observed, T - lag), counts per (origin,dest,time).
    row_index : pd.MultiIndex
        (origin_zone, dest_zone) for M rows.
    time_cols : pd.Index
        Destination time slices (df.columns[lag:]).
    """
    if lag <= 0:
        raise ValueError("lag must be a positive integer")

    n_agents, T = df.shape
    TT = T - lag
    if TT <= 0:
        # nothing to count
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["origin", "dest"])
        return csr_matrix((0, 0)), empty_idx, df.columns[lag:]

    # -------- Pass 1: build zone vocabulary without flattening everything --------
    zones_set = set()
    # iterate columns to avoid creating one huge flattened array
    for c in df.columns:
        zones_set.update(pd.unique(df[c].to_numpy()))
    zones = pd.Index(sorted(zones_set))
    K = len(zones)

    # Helpers
    def codes_of(values):
        """Map zone ids (1D array) -> dense codes [0..K-1]."""
        return zones.get_indexer(values)  # -1 indicates missing (shouldn't happen here)

    # -------- Pass 2: stream over time to build sparse matrix for observed pairs only --------
    pair2row = {}                 # map flattened pair (oi*K + di) -> compact row id
    rows_acc, cols_acc, dat_acc = [], [], []
    next_row_id = 0

    # Preallocate reusable time index vector for speed
    time_cols = df.columns[lag:]

    for t in range(TT):
        # Origins at t, destinations at t+lag
        o = df.iloc[:, t].to_numpy()
        d = df.iloc[:, t + lag].to_numpy()

        if not include_self:
            mask = (o != d)
            if not np.any(mask):
                continue
            o = o[mask]; d = d[mask]

        oi = codes_of(o)
        di = codes_of(d)

        # (Should be all >=0; guard anyway)
        valid = (oi >= 0) & (di >= 0)
        if not np.any(valid):
            continue
        oi = oi[valid]; di = di[valid]

        # Encode (origin,dest) into a flat key and aggregate counts for this time
        pair_key = oi.astype(np.int64) * np.int64(K) + di.astype(np.int64)
        uniq_pairs, counts = np.unique(pair_key, return_counts=True)

        # Map each unique pair to a compact row id (create new rows as needed)
        for pk, cnt in zip(uniq_pairs, counts):
            rid = pair2row.get(pk)
            if rid is None:
                rid = next_row_id
                pair2row[pk] = rid
                next_row_id += 1
            rows_acc.append(rid)
            cols_acc.append(t)
            dat_acc.append(int(cnt))

    # Build sparse matrix from accumulated triplets
    if len(dat_acc) == 0:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["origin", "dest"])
        return csr_matrix((0, TT)), empty_idx, time_cols

    M = coo_matrix((np.array(dat_acc, dtype=np.int64),
                    (np.array(rows_acc, dtype=np.int64),
                     np.array(cols_acc, dtype=np.int64))),
                   shape=(next_row_id, TT)).tocsr()

    # Recover (origin, dest) labels from pair2row
    # Invert the mapping to build aligned arrays
    inv = sorted(pair2row.items(), key=lambda kv: kv[1])  # sort by row id
    pair_flat = np.array([kv[0] for kv in inv], dtype=np.int64)
    ori_codes = (pair_flat // K).astype(int)
    des_codes = (pair_flat %  K).astype(int)
    row_index = pd.MultiIndex.from_arrays(
        [zones.take(ori_codes).to_numpy(), zones.take(des_codes).to_numpy()],
        names=["origin", "dest"]
    )

    return M, row_index, time_cols


def reorder_rows_within_origin_by_totals(M: csr_matrix, row_index: pd.MultiIndex,
                                         origin_asc=True, totals_desc=True):
    """
    Reorder rows of a sparse counts matrix M (rows=(O,D), cols=time)
    so that within each origin O, destinations D are sorted by total transitions
    (row sum across time). By default: origin ascending, totals descending.

    Returns:
      M_reordered (csr_matrix), row_index_reordered (MultiIndex)
    """
    if not isinstance(row_index, pd.MultiIndex) or row_index.nlevels != 2:
        raise ValueError("row_index must be a MultiIndex with levels ['origin','dest']")

    # Row totals (sum over time)
    totals = np.asarray(M.sum(axis=1)).ravel()
    # Build a small DataFrame to sort rows
    ord_df = pd.DataFrame({
        'origin': row_index.get_level_values(0).to_numpy(),
        'dest':   row_index.get_level_values(1).to_numpy(),
        'total':  totals
    })

    # Sort: origin (asc/desc), then total (desc/asc)
    ord_df_sorted = ord_df.sort_values(
        by=['origin', 'total'],
        ascending=[origin_asc, not totals_desc],
        kind='mergesort'  # stable within groups
    )
    new_order = ord_df_sorted.index.to_numpy()

    # Reindex rows of the sparse matrix
    M_reordered = M[new_order, :]
    # Reindex the MultiIndex
    row_index_reordered = row_index[new_order]

    return M_reordered, row_index_reordered

def zone_occupancy_counts(df):
    """
    For each time slice (column), count how many agents are in each zone.
    Returns a DataFrame with rows = zone ids (sorted ascending),
    cols = time slices (same order as df.columns),
    values = counts of agents in that zone at that time.
    """
    # value_counts per column, then concat along columns
    counts = pd.concat({c: df[c].value_counts() for c in df.columns}, axis=1)
    # ensure all time columns present and ordered like input
    counts = counts.reindex(columns=df.columns)
    # fill missing with 0, sort rows by zone id
    counts = counts.fillna(0).astype(int).sort_index()
    return counts


def binomial_cascade_sizes_fast(transition_counts: pd.DataFrame,
                                occupancy_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized binomial-cascade sizes without groupby/apply.
    Rows of transition_counts must be MultiIndex (origin, dest),
    already ordered by desired dest order WITHIN each origin.

    occupancy_counts: index=origin, columns=time at t.
    transition_counts: rows=(origin,dest), columns=times at t+1.

    Returns a DataFrame same shape/index/columns as transition_counts
    with sizes = occupancy(t) - cumulative(previous transitions)(t+1), clipped at 0.
    """
    # ----- Align occupancy columns vs transitions -----
    T_tr = transition_counts.shape[1]
    T_occ = occupancy_counts.shape[1]
    if T_occ == T_tr + 1:
        occ_aligned = occupancy_counts.iloc[:, :-1]
    elif T_occ == T_tr:
        occ_aligned = occupancy_counts
    else:
        raise ValueError("occupancy_counts must have same #cols as transitions OR exactly one more (use t for t+1).")

    # ----- Extract arrays -----
    C = transition_counts.to_numpy(copy=False)        # shape (R, T), R = #rows (origin,dest)
    idx = transition_counts.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
        raise ValueError("transition_counts rows must be a MultiIndex (origin,dest).")

    # Codes give integer labels per level in the order they appear in .levels
    origin_codes = idx.codes[0] if hasattr(idx, 'codes') else idx.codes   # older pandas: .labels
    # Find block boundaries where origin changes
    # positions where origin_code changes, plus start/end
    change = np.flatnonzero(np.diff(origin_codes)) + 1
    starts = np.r_[0, change]
    ends   = np.r_[change, len(origin_codes)]

    # Unique origins in the same order as blocks
    origins_in_rows = idx.get_level_values(0).to_numpy()
    unique_origins_in_order = origins_in_rows[starts]

    # Build occupancy matrix aligned to those origins (fill missing with zeros)
    occ_mat = occ_aligned.reindex(index=unique_origins_in_order).fillna(0.0).to_numpy(dtype=np.float64)

    # Output buffer
    S = np.empty_like(C, dtype=np.int64)

    # ----- Process each origin block in pure NumPy -----
    # For each origin block C_block (nD x T):
    # sizes = occ_row - cumsum(previous rows); use cumsum with prepend=0 to avoid extra alloc
    for bi, (s, e) in enumerate(zip(starts, ends)):
        C_block = C[s:e, :]  # (nD x T)
        if C_block.size == 0:
            continue

        # occupancy row for this origin (length T)
        N = occ_mat[bi, :]   # (T,)

        # cum_prev[j,:] = sum_{k<j} C_block[k,:]
        # np.cumsum with prepend adds a zero row efficiently
        cum_prev = np.cumsum(C_block[:-1, :], axis=0)
        # allocate sizes block and compute
        # first row: N - 0
        S[s, :] = N
        # remaining rows: N - cum_prev[j-1]
        if C_block.shape[0] > 1:
            S[s+1:e, :] = N - cum_prev

        # clip to non-negative (in case of minor inconsistencies)
        np.maximum(S[s:e, :], 0, out=S[s:e, :])

    # Wrap back to DataFrame
    sizes_df = pd.DataFrame(S, index=transition_counts.index, columns=transition_counts.columns)
    return sizes_df


# occ = pd.DataFrame({
#     't0': [2,1,0], 't1': [1,2,1], 't2': [0,1,3], 't3':[1,0,3]
# }, index=[1,2,3])  # origins=1,2,3

# # transitions ending at t1..t3 (lag=1), rows ordered per O by your rule
# tc = pd.DataFrame(
#     # rows: (O,D) pairs
#     data=[[1,0,0],  # 1->2 at t1
#           [1,1,0],  # 1->3 at t1,t2
#           [0,1,0],  # 2->3 at t2
#           [0,0,1]], # 3->1 at t3
#     index=pd.MultiIndex.from_tuples([(1,2),(1,3),(2,3),(3,1)], names=['origin','dest']),
#     columns=['t1','t2','t3']
# )

# sizes = binomial_cascade_sizes_fast(tc, occ)
# print(sizes)
# def filter_by_prefix_threshold(M: scipy.sparse._csr.csr_matrix,
#                                row_index: pd.core.indexes.multi.MultiIndex,
#                                transition_counts: pd.DataFrame,
#                                size_counts: pd.DataFrame,
#                                occupancy_counts: pd.DataFrame | pd.Series,
#                                threshold: float = 0.99):
#     """
#     Filter transition_counts and size_counts to only keep (origin, dest) pairs
#     whose cumulative counts (per origin) cover up to `threshold` of occupancy.

#     Special handling:
#     - If an origin has fewer than two (origin, dest) rows (i.e., only one direction),
#       do NOT filter that origin (keep all its rows).

#     Parameters
#     ----------
#     transition_counts : DataFrame
#         MultiIndex (origin, dest) on rows, time columns on columns.
#     size_counts : DataFrame
#         Same shape/index as transition_counts.
#     occupancy_counts : DataFrame or Series
#         Occupancy counts per origin (index = origin).
#     threshold : float, default 0.99
#         Coverage threshold for cumulative proportion.

#     Returns
#     -------
#     (DataFrame, DataFrame)
#         Filtered transition_counts and size_counts.
#     """

#     # pooled totals over time
#     pooled_transition_counts = (
#         pd.Series(np.array(M.sum(axis=1)).flatten(),
#                   index=row_index)
#         .reset_index()
#         .rename(columns={0: "count"})
#     )

#     # allow Series or single-column DataFrame for occupancy
#     if isinstance(occupancy_counts, pd.DataFrame):
#         pooled_occupancy_counts = pd.Series(
#             occupancy_counts.values.sum(axis=1),
#             index=occupancy_counts.index
#         )
#     else:
#         pooled_occupancy_counts = occupancy_counts

#     def prefix_until_threshold(g, occ, threshold=0.99):
#         """
#         Keep rows in group g until cumulative fraction of occupancy reaches threshold.
#         If group has < 2 rows (only one direction), return g unchanged (no filtering).
#         """
#         # Need at least two directions; otherwise skip filtering
#         if len(g) < 2:
#             return g.assign(cum_frac=(g["count"] / (occ if occ else 1)).cumsum())  # harmless annotation

#         # Guard against zero occupancy
#         denom = occ if occ and occ != 0 else 1.0

#         frac = g["count"].cumsum() / denom
#         mask = frac >= threshold
#         if mask.any():
#             s = mask.idxmax()  # first index reaching/exceeding threshold
#             return g.loc[:s].assign(cum_frac=frac.loc[:s])
#         else:
#             return g.assign(cum_frac=frac)

#     # For per-origin grouping, we need to ensure rows are in a deterministic order.
#     # We preserve the current order of pooled_transition_counts; if you prefer
#     # "largest counts first", sort inside the group before cumsum.
#     result_df = (
#         pooled_transition_counts
#         .groupby("origin", group_keys=False, sort=False)
#         .apply(lambda g: prefix_until_threshold(
#             g, pooled_occupancy_counts.loc[g.name], threshold=threshold
#         ))
#     )

#     # Build MultiIndex of kept (origin, dest) pairs
#     pairs = (
#         result_df[["origin", "dest"]]
#         .drop_duplicates()
#         .set_index(["origin", "dest"])
#         .index
#     )

#     # Filter both DataFrames (keep intersection only)
#     pairs_tc = transition_counts.index.intersection(pairs)
#     pairs_sc = size_counts.index.intersection(pairs)

#     return transition_counts.loc[pairs_tc], size_counts.loc[pairs_sc]

def make_individual_transition_matrix(
    agent_locations: pd.Series,              # DatetimeIndex (length T+1) → zone ids at each time slice
    include_self: bool = True,               # include (i→i) transitions?
    threshold: float = 0.95,                 # keep smallest prefix per origin whose mass ≥ threshold
    origin_asc: bool = True,                 # ordering of origin groups in the output
    totals_desc: bool = True,                # order destinations within each origin by descending totals
    min_origin_count: int = 2,               # minimum # of transitions required for an origin to be kept
    inactive_as_nan: bool = False             # if True, keep NaN when origin inactive at a time slice
) -> pd.DataFrame:
    """
    Build an ordered, filtered timewise transition count matrix for a single agent trajectory.

    Returns
    -------
    transition_counts : DataFrame
        Rows = MultiIndex (origin=i, dest=j), Columns = time slices = times[1:].
        For each origin, destinations are sorted by total counts (desc by default),
        then truncated to the smallest prefix whose cumulative mass ≥ threshold.
        Origins with fewer than `min_origin_count` transitions are dropped entirely.

        If `inactive_as_nan` is True:
            - For a given column (time t), rows for origins that were NOT active at t-1 remain NaN.
            - For the active origin at t-1, the chosen dest is 1 and all other dests are 0.
        If `inactive_as_nan` is False:
            - Unobserved entries are filled with 0 everywhere (legacy behavior).
    """

    # --- 0) Basic checks & prep
    if not isinstance(agent_locations.index, pd.DatetimeIndex):
        raise ValueError("agent_locations must have a DatetimeIndex")

    # Drop NaNs but keep alignment
    aloc = agent_locations.dropna()
    if len(aloc) < 2:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["origin", "dest"]),
            columns=pd.DatetimeIndex([], name=agent_locations.index.name)
        )

    times   = aloc.index
    origins_full = aloc.values[:-1]
    dests_full   = aloc.values[1:]
    # Destination timestamps = times[1:]
    dest_time_cols_full = pd.to_datetime(times[1:])

    # Optionally exclude self transitions
    mask = np.ones_like(origins_full, dtype=bool)
    if not include_self:
        mask = origins_full != dests_full

    origins = origins_full[mask]
    dests   = dests_full[mask]
    tcols   = times[1:][mask]  # destination times associated with each transition

    if len(tcols) == 0:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["origin", "dest"]),
            columns=dest_time_cols_full
        )

    # --- 1) Build transition rows
    df_trans = pd.DataFrame({
        "origin": origins,
        "dest":   dests,
        "time":   pd.to_datetime(tcols),
        "cnt":    1.0
    })

    # --- 1.5) Remove rare origins
    origin_counts = df_trans.groupby("origin").size()
    good_origins = origin_counts[origin_counts >= min_origin_count].index
    df_trans = df_trans[df_trans["origin"].isin(good_origins)]

    if df_trans.empty:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["origin", "dest"]),
            columns=dest_time_cols_full
        )

    # --- 2) Pivot into (origin,dest) × time matrix (no fill here to allow NaN handling below)
    transition_counts = (
        df_trans.pivot_table(
            index=["origin", "dest"],
            columns="time",
            values="cnt",
            aggfunc="sum"
        )
    )

    # Ensure full coverage of all destination times
    if inactive_as_nan:
        # Keep NaN for unobserved cells (inactive origins at a time)
        transition_counts = transition_counts.reindex(columns=dest_time_cols_full)
    else:
        # Legacy: fill missing with 0 everywhere
        transition_counts = transition_counts.reindex(columns=dest_time_cols_full, fill_value=0.0)

    transition_counts = transition_counts.sort_index(axis=1)

    # If inactive_as_nan=True, we need to ensure one-hot within the active origin only,
    # and keep other origins as NaN for that column.
    if inactive_as_nan:
        # origin present at time t-1 for each destination column t (based on *unmasked* series)
        origin_at_tminus1 = pd.Series(origins_full, index=dest_time_cols_full)
        # Only columns that survived (intersection) need processing
        cols_to_process = transition_counts.columns.intersection(origin_at_tminus1.index)
        if len(cols_to_process) > 0:
            row_index = transition_counts.index
            origins_level = row_index.get_level_values("origin")
            # Fill NaN with 0 ONLY for the active origin in each column
            for t in cols_to_process:
                o = origin_at_tminus1.loc[t]
                mask_rows = (origins_level == o)
                # fill NaN -> 0 for that origin's rows at this column; leave other origins as NaN
                transition_counts.loc[mask_rows, t] = transition_counts.loc[mask_rows, t].fillna(0.0)

    # --- 3) Sort destinations within each origin by total counts
    # Note: sum() skips NaN by default, which is desired
    totals_per_row = transition_counts.sum(axis=1)
    tc = transition_counts.copy()
    tc["__totals__"] = totals_per_row
    tc = (
        tc.sort_values(["origin", "__totals__"],
                       ascending=[origin_asc, not totals_desc])
          .drop(columns="__totals__")
    )

    # --- 4) Apply cumulative mass threshold within each origin
    totals = tc.sum(axis=1)
    kept_idx = []
    for origin, grp in totals.groupby(level=0, sort=False):
        gvals = grp.values
        gidx  = grp.index
        total_mass = gvals.sum()
        if total_mass <= 0:
            continue
        csum = np.cumsum(gvals / total_mass)
        k = np.searchsorted(csum, threshold, side="left") + 1
        kept_idx.append(gidx[:k])

    if len(kept_idx) == 0:
        return tc.iloc[0:0, :]

    kept_idx = reduce(lambda a, b: a.append(b), kept_idx)
    tc_filtered = tc.loc[kept_idx]

    # --- 5) Finalize index and columns
    tc_filtered.index = pd.MultiIndex.from_tuples(tc_filtered.index.tolist(), names=["origin", "dest"])
    tc_filtered.columns = pd.DatetimeIndex(tc_filtered.columns, name=agent_locations.index.name)

    return tc_filtered

