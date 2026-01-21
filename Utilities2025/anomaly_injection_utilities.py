import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

def _runs(x: np.ndarray) -> List[Tuple[int, int, Union[int, str]]]:
    """Consecutive runs as (start, end_exclusive, value)."""
    if x.size == 0:
        return []
    b = np.nonzero(np.r_[True, x[1:] != x[:-1]])[0]
    starts = b
    ends = np.r_[b[1:], len(x)]
    vals = x[starts]
    return [(int(s), int(e), vals[i]) for i, (s, e) in enumerate(zip(starts, ends))]

def _all_sps_in_day(
    x: np.ndarray, day_start: int, day_end: int, staying_zones: set, min_consec: int
) -> List[Tuple[int,int,int,int,int,int,Union[int,str],Union[int,str]]]:
    """
    Return ALL S–P–S triplets fully inside [day_start, day_end).
    Each item: (s1_s, s1_e, p_s, p_e, s2_s, s2_e, z1, z2).
    """
    out = []
    runs = _runs(x[day_start:day_end])
    runs = [(s+day_start, e+day_start, z) for s, e, z in runs]  # global indices
    stay_runs = [(s,e,z) for s,e,z in runs if (z in staying_zones and (e-s) >= min_consec)]
    # For each staying run, pair to the next staying run to form an S–P–S
    for i in range(len(stay_runs)-1):
        s1_s, s1_e, z1 = stay_runs[i]
        for j in range(i+1, len(stay_runs)):
            s2_s, s2_e, z2 = stay_runs[j]
            p_s, p_e = s1_e, s2_s
            if p_s >= p_e:
                continue
            # require the whole triplet inside the day
            if s1_s >= day_start and s2_e <= day_end:
                out.append((s1_s, s1_e, p_s, p_e, s2_s, s2_e, z1, z2))
            break  # only the next staying run can be the destination for this S1
    return out

def inject_daily_sps_shift(
    traj: pd.Series,
    win_min: int,
    win_max: int,
    *,
    day_len: int = 288,
    min_consec: int = 12,
    mode: str = "random",        # 'early' | 'late' | 'random'
    max_shift: int = 48,
    sps_pick: str = "random",    # 'first' | 'random' | 'k'
    sps_k: Optional[int] = None, # 0-based index if sps_pick='k'
    rng: Optional[Union[int, np.random.Generator]] = None
) -> Dict[str, Union[pd.Series, List[Dict], Dict]]:
    """
    For each day in [win_min, win_max), pick ONE S–P–S (not necessarily the first)
    and shift the path earlier or later:

      EARLY: shift path left by d (<= max_shift and <= origin-stay length);
             fill freed tail [new_P_right, old_P_end) with destination zone (arrive early).
      LATE:  shift path right by d (<= max_shift and <= destination-stay head length);
             fill head gap [old_P_start, old_P_start+d) with origin zone (depart late).

    Selection of S–P–S per day:
      - sps_pick='random'  → choose uniformly among all S–P–S in the day.
      - sps_pick='first'   → choose the first (previous behavior).
      - sps_pick='k'       → choose the k-th S–P–S if it exists (else skip).

    Length-preserving; stays within day; single triplet per day.
    """
    assert 0 <= win_min < win_max <= len(traj), "Window out of range"
    x0 = traj.to_numpy()
    x = x0.copy()
    n = len(x0)
    if isinstance(rng, (int, type(None))):
        rng = np.random.default_rng(rng)

    # Identify staying zones globally
    runs_all = _runs(x0)
    staying_zones = {z for s,e,z in runs_all if (e-s) >= min_consec}

    logs: List[Dict] = []
    changed_idxs: List[int] = []

    day_start = (win_min // day_len) * day_len
    day_end = ((win_max - 1) // day_len) * day_len + day_len

    for d0 in range(day_start, day_end, day_len):
        if d0 >= win_max:
            break
        d1 = min(d0 + day_len, n)
        if d1 <= win_min:
            continue

        # collect ALL S–P–S within the day
        trips = _all_sps_in_day(x0, d0, d1, staying_zones, min_consec)
        if not trips:
            continue

        # pick which S–P–S to modify
        if sps_pick == "first":
            trip = trips[0]
            trip_idx = 0
        elif sps_pick == "k" and sps_k is not None and 0 <= sps_k < len(trips):
            trip = trips[sps_k]
            trip_idx = sps_k
        elif sps_pick == "random":
            trip_idx = int(rng.integers(0, len(trips)))
            trip = trips[trip_idx]
        else:
            # invalid k or sps_pick → skip this day
            continue

        s1_s, s1_e, p_s, p_e, s2_s, s2_e, z1, z2 = trip
        P_len = p_e - p_s
        L1 = s1_e - s1_s
        L2 = s2_e - s2_s
        if P_len <= 0:
            continue

        # early/late decision
        if mode == "early":
            do_early = True
        elif mode == "late":
            do_early = False
        else:  # random
            do_early = bool(rng.integers(0, 2))

        if do_early:
            dmax = min(max_shift, L1)
            if dmax <= 0:
                continue
            d = int(rng.integers(1, dmax + 1))
            left = p_s - d
            right = p_e - d
            if left < s1_s:
                d -= (s1_s - left)
                if d <= 0:
                    continue
                left, right = p_s - d, p_e - d
            path_vals = x0[p_s:p_e].copy()
            before = x[left:right].copy()
            x[left:right] = path_vals
            # fill the freed tail with destination zone
            fill_start, fill_end = right, p_e
            before_tail = x[fill_start:fill_end].copy()
            x[fill_start:fill_end] = z2

            ch1 = np.where(x[left:right] != before)[0] + left
            ch2 = np.where(x[fill_start:fill_end] != before_tail)[0] + fill_start
            changed_idxs.extend(ch1.tolist()); changed_idxs.extend(ch2.tolist())

            logs.append({
                "day_start": d0, "trip_index": trip_idx, "mode": "early", "shift": d,
                "origin_zone": z1, "dest_zone": z2,
                "orig_S1": (s1_s, s1_e), "orig_P": (p_s, p_e), "orig_S2": (s2_s, s2_e),
                "new_P": (left, right), "filled_dest": (fill_start, fill_end)
            })
        else:
            dmax = min(max_shift, L2)
            if dmax <= 0:
                continue
            d = int(rng.integers(int(dmax*0.7), dmax + 1))
            left = p_s + d
            right = p_e + d
            if right > s2_e:
                d -= (right - s2_e)
                if d <= 0:
                    continue
                left, right = p_s + d, p_e + d
            # fill head gap with origin zone
            fill_start, fill_end = p_s, p_s + d
            before_head = x[fill_start:fill_end].copy()
            x[fill_start:fill_end] = z1
            # move path later
            path_vals = x0[p_s:p_e].copy()
            before = x[left:right].copy()
            x[left:right] = path_vals

            ch1 = np.where(x[fill_start:fill_end] != before_head)[0] + fill_start
            ch2 = np.where(x[left:right] != before)[0] + left
            changed_idxs.extend(ch1.tolist()); changed_idxs.extend(ch2.tolist())

            logs.append({
                "day_start": d0, "trip_index": trip_idx, "mode": "late", "shift": d,
                "origin_zone": z1, "dest_zone": z2,
                "orig_S1": (s1_s, s1_e), "orig_P": (p_s, p_e), "orig_S2": (s2_s, s2_e),
                "new_P": (left, right), "filled_origin": (fill_start, fill_end)
            })

    # merge changed indices into intervals
    def _merge(idxs: List[int]) -> List[Tuple[int,int]]:
        if not idxs: return []
        idxs = sorted(set(int(i) for i in idxs))
        out = []
        s = prev = idxs[0]
        for k in idxs[1:]:
            if k != prev + 1:
                out.append((s, prev+1))
                s = k
            prev = k
        out.append((s, prev+1))
        return out

    intervals = _merge(changed_idxs)
    return {
        "traj_new": pd.Series(x, index=traj.index, name=traj.name),
        "changed_intervals": intervals,
        "log": logs,
        "meta": {
            "win": (win_min, win_max),
            "day_len": day_len,
            "min_consec": min_consec,
            "sps_pick": sps_pick,
            "sps_k": sps_k
        }
    }


def swap_days(
    input_traj: pd.Series,
    win_min: int,
    win_max: int,
    *,
    day_len: int = 288,
    rng: Optional[Union[int, np.random.Generator]] = None
) -> Tuple[pd.Series, List[Tuple[int, int]]]:
    """
    Randomly swap whole days before win_min with the consecutive days in [win_min, win_max).
    Reshapes to D x day_len internally and flattens back to a 1D Series.

    Returns:
      flat_traj: flattened Series after swapping (index preserved)
      swapped_pairs: list of (source_day, target_day) pairs swapped
    """
    # basic checks
    n = len(input_traj)
    assert n % day_len == 0, "Trajectory length must be a multiple of day_len."
    assert 0 <= win_min < win_max <= n, "Window out of range."
    assert win_min % day_len == 0 and win_max % day_len == 0, "win_min/max must align with day boundaries."

    # derive day indices
    D = n // day_len
    start_day = win_min // day_len
    end_day   = win_max // day_len
    val_days  = end_day - start_day
    assert start_day >= val_days, "Not enough earlier days to sample from."

    # RNG
    if isinstance(rng, (int, type(None))):
        rng = np.random.default_rng(rng)

    # --- reshape to D x day_len (integrated reshape_traj_to_day_matrix)
    values = input_traj.to_numpy().reshape(D, day_len)
    daily_traj = pd.DataFrame(values, index=np.arange(D), columns=np.arange(day_len))

    # sample and target day indices
    src_days = rng.choice(np.arange(start_day), size=val_days, replace=False).tolist()
    tgt_days = list(range(start_day, end_day))

    # swap rows in-place on the DataFrame
    swapped_pairs: List[Tuple[int, int]] = []
    for s, t in zip(src_days, tgt_days):
        daily_traj.iloc[[s, t]] = daily_traj.iloc[[t, s]].values
        swapped_pairs.append((int(s), int(t)))

    # flatten back to series, preserving original index & name
    flat_traj = pd.Series(daily_traj.to_numpy().reshape(-1), index=input_traj.index, name=input_traj.name)
    return flat_traj, swapped_pairs

def inject_anomaly(
    agent_series,
    test_st,
    test_end,
    threshold=0.175,
    max_shift=60,
    max_iter=1000,
    sps_pick="random",
    mode="random",
    verbose=True
):
    """
    Keep injecting anomalies until the proportion of changed trajectory 
    exceeds a given threshold or max_iter is reached.

    Parameters
    ----------
    agent_series : pd.Series
        Original trajectory data (e.g., zone IDs).
    test_st, test_end : int
        Start and end indices for the injection period.
    anomaly_obj : object
        Object with .inject_daily_sps_shift() method.
    threshold : float, default 0.15
        Minimum proportion of changed trajectory required.
    max_shift : int, default 60
        Max shift passed to injection function.
    max_iter : int, default 1000
        Safety cap for the number of attempts.
    sps_pick, mode : str
        Arguments passed to inject_daily_sps_shift().
    verbose : bool, default True
        Whether to print progress.

    Returns
    -------
    dict
        The injection result from the last iteration.
    float
        Final proportion of changed points.
    int
        Number of iterations used.
    """
    prop = 0.0
    i = 0
    while prop <= threshold and i < max_iter:
        res = inject_daily_sps_shift(
            agent_series,
            test_st,
            test_end,
            sps_pick=sps_pick,
            mode=mode,
            max_shift=max_shift
        )
        prop = np.mean(
            res['traj_new'].iloc[test_st:test_end] !=
            agent_series.iloc[test_st:test_end]
        )
        i += 1
    if verbose:
        print(f"Stopped after {i} iterations — final proportion: {prop:.4f}")
        
    res['success'] = prop >= threshold
    return res

def find_lbf_threshold(
    logbf_2col: np.ndarray,
    *,
    plot: bool = False,
    max_fpr: float | None = None,   # e.g., 0.01 for 1% FPR cap
    beta: float = 0.5               # F_beta with beta<1 emphasizes precision
):
    """
    Sweep threshold τ from high→low (smaller = more anomalous, predict if s <= τ),
    and find τ that maximizes F_beta under an optional FPR cap.

    Parameters
    ----------
    logbf_2col : array (N, 2)
        [:,0] = no-anomaly scores (negatives)
        [:,1] = anomaly scores (positives)
    plot : bool
        Plot F_beta / Precision / Recall / FPR / Accuracy vs τ if True.
    max_fpr : float | None
        If set, only thresholds with FPR ≤ max_fpr are eligible.
    beta : float
        F_beta parameter. beta<1 emphasizes precision; beta>1 emphasizes recall.

    Returns
    -------
    best : dict
        {'tau','fbeta','precision','recall','fpr','accuracy','support':(num_pos,num_neg),'feasible':bool}
    curves : dict
        {'tau','precision','recall','fbeta','fpr','accuracy'}
    """
    assert logbf_2col.ndim == 2 and logbf_2col.shape[1] == 2

    s_neg = logbf_2col[:, 0].astype(float)  # no anomaly
    s_pos = logbf_2col[:, 1].astype(float)  # anomaly

    # keep only finite
    s_neg = s_neg[np.isfinite(s_neg)]
    s_pos = s_pos[np.isfinite(s_pos)]

    y = np.concatenate([np.ones_like(s_pos, dtype=int), np.zeros_like(s_neg, dtype=int)])
    s = np.concatenate([s_pos, s_neg])

    uniq = np.unique(s)
    taus = np.concatenate([[np.inf], uniq[::-1], [-np.inf]])  # sweep high→low

    P = int((y == 1).sum())
    N = int((y == 0).sum())

    tau_curve, prec_curve, rec_curve, fbeta_curve, fpr_curve, acc_curve = [], [], [], [], [], []

    def fbeta(prec, rec, beta):
        if prec + rec == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * (prec * rec) / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0

    best = {
        'tau': None, 'fbeta': -1.0, 'precision': 0.0, 'recall': 0.0,
        'fpr': 1.0, 'accuracy': 0.0, 'support': (P, N), 'feasible': True
    }
    # fallback if no τ meets FPR cap: choose minimal FPR, tie-break by higher F_beta
    best_min_fpr = {
        'tau': None, 'fbeta': -1.0, 'precision': 0.0, 'recall': 0.0,
        'fpr': 1.0, 'accuracy': 0.0
    }

    for tau in taus:
        yhat = (s <= tau).astype(int)  # smaller score -> more anomalous (positive)

        TP = np.sum((yhat == 1) & (y == 1))
        FP = np.sum((yhat == 1) & (y == 0))
        FN = np.sum((yhat == 0) & (y == 1))
        TN = np.sum((yhat == 0) & (y == 0))

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        acc  = (TP + TN) / (P + N) if (P + N) > 0 else 0.0
        f_b  = fbeta(prec, rec, beta)

        tau_curve.append(float(tau))
        prec_curve.append(float(prec))
        rec_curve.append(float(rec))
        fpr_curve.append(float(fpr))
        acc_curve.append(float(acc))
        fbeta_curve.append(float(f_b))

        # track minimal-FPR candidate (fallback)
        if (best_min_fpr['tau'] is None) or (fpr < best_min_fpr['fpr']) or \
           (np.isclose(fpr, best_min_fpr['fpr']) and (f_b > best_min_fpr['fbeta'])):
            best_min_fpr.update({
                'tau': float(tau), 'fbeta': float(f_b), 'precision': float(prec),
                'recall': float(rec), 'fpr': float(fpr), 'accuracy': float(acc)
            })

        # enforce FPR cap if provided
        if (max_fpr is not None) and (fpr > max_fpr):
            continue

        # choose τ that maximizes F_beta; tie-break: larger τ (stricter) to favor precision
        if (f_b > best['fbeta']) or (np.isclose(f_b, best['fbeta']) and (best['tau'] is None or tau > best['tau'])):
            best.update({
                'tau': float(tau), 'fbeta': float(f_b), 'precision': float(prec),
                'recall': float(rec), 'fpr': float(fpr), 'accuracy': float(acc)
            })

    if (max_fpr is not None) and (best['tau'] is None):
        best = {**best_min_fpr, 'support': (P, N), 'feasible': False}
    else:
        best['support'] = (P, N)
        best['feasible'] = True

    curves = {
        'tau': np.array(tau_curve),
        'precision': np.array(prec_curve),
        'recall': np.array(rec_curve),
        'fbeta': np.array(fbeta_curve),
        'fpr': np.array(fpr_curve),
        'accuracy': np.array(acc_curve),
    }

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(curves['tau'], curves['fbeta'], label=f'F$_{{{beta}}}$')
        plt.plot(curves['tau'], curves['precision'], '--', label='Precision')
        plt.plot(curves['tau'], curves['recall'], '--', label='Recall')
        plt.plot(curves['tau'], curves['fpr'], ':', label='FPR')
        plt.plot(curves['tau'], curves['accuracy'], '-.', label='Accuracy')
        if best['tau'] is not None:
            plt.axvline(best['tau'], color='k', linestyle=':', label=f"Best τ = {best['tau']:.3f}")
        if max_fpr is not None:
            plt.axhline(max_fpr, color='r', linestyle='--', alpha=0.6, label=f"FPR cap = {max_fpr:.3f}")
        plt.xlabel('Threshold (τ, log BF)  [smaller = more anomalous]')
        plt.ylabel('Score')
        plt.title(f'F$_{{{beta}}}$ / Precision / Recall / FPR / Accuracy vs Threshold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return best, curves

def find_lbf_threshold2(
    logbf_2col: np.ndarray,
    *,
    plot: bool = False,
    max_fpr: float | None = None    # If set, we only consider τ with FPR ≤ max_fpr
):
    """
    Sweep threshold τ from high→low (smaller = more anomalous; predict anomaly if s <= τ),
    and choose τ that maximizes Recall (Power) subject to an optional FPR cap.

    Parameters
    ----------
    logbf_2col : array (N, 2)
        [:,0] = no-anomaly scores (negatives / regular)
        [:,1] = anomaly scores (positives)
    plot : bool
        Plot Recall / FPR / Precision vs τ if True.
    max_fpr : float | None
        If set (e.g., 0.10), only thresholds with FPR ≤ max_fpr are eligible.

    Returns
    -------
    best : dict
        {
          'tau', 'recall', 'fpr', 'precision',
          'support': (num_pos, num_neg),
          'feasible': bool   # False if no τ satisfies the FPR cap; best then is min-FPR fallback
        }
    curves : dict
        {'tau','precision','recall','fpr'}
    """
    assert logbf_2col.ndim == 2 and logbf_2col.shape[1] == 2

    s_neg = logbf_2col[:, 0].astype(float)  # regular
    s_pos = logbf_2col[:, 1].astype(float)  # anomaly

    # Keep only finite
    s_neg = s_neg[np.isfinite(s_neg)]
    s_pos = s_pos[np.isfinite(s_pos)]

    # Labels and scores
    y = np.concatenate([np.ones_like(s_pos, dtype=int), np.zeros_like(s_neg, dtype=int)])
    s = np.concatenate([s_pos, s_neg])

    # Threshold sweep: high -> low
    uniq = np.unique(s)
    taus = np.concatenate([[np.inf], uniq[::-1], [-np.inf]])

    P = int((y == 1).sum())
    N = int((y == 0).sum())

    tau_curve, prec_curve, rec_curve, fpr_curve = [], [], [], []

    # Initialize best candidate (for feasible set) and a fallback at minimal FPR
    best = {'tau': None, 'recall': -1.0, 'fpr': 1.0, 'precision': 0.0, 'support': (P, N), 'feasible': True}
    best_min_fpr = {'tau': None, 'recall': 0.0, 'fpr': 1.0, 'precision': 0.0}

    # Small tolerance for floats
    atol = 1e-12

    for tau in taus:
        # Decision rule: smaller score => more anomalous
        yhat = (s <= tau).astype(int)

        TP = int(np.sum((yhat == 1) & (y == 1)))
        FP = int(np.sum((yhat == 1) & (y == 0)))
        FN = int(np.sum((yhat == 0) & (y == 1)))
        TN = int(np.sum((yhat == 0) & (y == 0)))

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        tau_curve.append(float(tau))
        prec_curve.append(float(prec))
        rec_curve.append(float(rec))
        fpr_curve.append(float(fpr))

        # Track minimal-FPR fallback (tie-break by higher recall, then higher precision)
        better_min_fpr = (fpr < best_min_fpr['fpr'] - atol) or \
                         (np.isclose(fpr, best_min_fpr['fpr'], atol=atol) and (rec > best_min_fpr['recall'] + atol)) or \
                         (np.isclose(fpr, best_min_fpr['fpr'], atol=atol) and np.isclose(rec, best_min_fpr['recall'], atol=atol) and (prec > best_min_fpr['precision'] + atol))
        if (best_min_fpr['tau'] is None) or better_min_fpr:
            best_min_fpr.update({'tau': float(tau), 'recall': float(rec), 'fpr': float(fpr), 'precision': float(prec)})

        # Enforce FPR cap if provided
        if (max_fpr is not None) and (fpr > max_fpr):
            continue

        # Among feasible τ, maximize recall; tie-break by lower FPR, then higher precision
        better = (rec > best['recall'] + atol) or \
                 (np.isclose(rec, best['recall'], atol=atol) and (fpr < best['fpr'] - atol)) or \
                 (np.isclose(rec, best['recall'], atol=atol) and np.isclose(fpr, best['fpr'], atol=atol) and (prec > best['precision'] + atol))
        if (best['tau'] is None) or better:
            best.update({'tau': float(tau), 'recall': float(rec), 'fpr': float(fpr), 'precision': float(prec)})

    # If no τ met the FPR cap, return minimal-FPR fallback and mark infeasible
    if (max_fpr is not None) and (best['tau'] is None):
        best = {**best_min_fpr, 'support': (P, N), 'feasible': False}
    else:
        best['support'] = (P, N)
        best['feasible'] = True

    curves = {
        'tau': np.array(tau_curve),
        'precision': np.array(prec_curve),
        'recall': np.array(rec_curve),
        'fpr': np.array(fpr_curve),
    }

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(curves['tau'], curves['recall'], label='Recall (Power)')
        plt.plot(curves['tau'], curves['fpr'], '--', label='FPR (Type I)')
        plt.plot(curves['tau'], curves['precision'], '-.', label='Precision')
        if best['tau'] is not None:
            plt.axvline(best['tau'], color='k', linestyle=':', label=f"Best τ = {best['tau']:.3f}")
        if max_fpr is not None:
            plt.axhline(max_fpr, color='r', linestyle='--', alpha=0.6, label=f"FPR cap = {max_fpr:.3f}")
        plt.xlabel('Threshold τ (log BF)   [smaller = more anomalous]')
        plt.ylabel('Metric value')
        plt.title('Recall / FPR / Precision vs Threshold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return best, curves
