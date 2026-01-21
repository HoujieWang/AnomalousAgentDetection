import Utilities2025.geo_utilities as mygeo
import pandas as pd 
import numpy as np
import os
import pyarrow
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import xticks
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from IPython.display import clear_output
from collections import deque
from sklearn.neighbors import BallTree
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

R = 6371008.8
def balltree_proximity_clusters(df, lat_col='lat', lon_col='lon', radius_m=200.0):
    """
    Build a sparse proximity graph with BallTree (Haversine), then return
    connected-component labels (spatial clusters) and the CSR adjacency.
    """
    # coords: degrees -> radians
    coords_deg = df[[lat_col, lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)
    eps_rad = float(radius_m) / R

    # BallTree radius neighbors (indices per point)
    tree = BallTree(coords_rad, metric='haversine')
    ind = tree.query_radius(coords_rad, r=eps_rad, return_distance=False)

    # Build sparse adjacency (0/1), then symmetrize + self-loops
    rows, cols = [], []
    for i, nbrs in enumerate(ind):
        for j in nbrs:
            rows.append(i); cols.append(j)
    n = len(coords_rad)
    data = np.ones(len(rows), dtype=np.uint8)
    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)  # make undirected
    G = G + csr_matrix((np.ones(n, dtype=np.uint8), (range(n), range(n))), shape=(n, n))

    # Connected components under the radius graph
    _, labels = connected_components(G, directed=False)
    return labels, G, coords_rad


def cap_components_with_bfs(G, base_labels, X=50):
    """
    Split any connected component (from base_labels) into contiguous chunks of size ≤ X,
    using BFS over the same adjacency G. Returns new compact cluster labels [0..K-1].
    """
    n = G.shape[0]
    new_labels = np.full(n, -1, dtype=int)
    next_cid = 0

    indptr, indices = G.indptr, G.indices

    for comp_id in np.unique(base_labels):
        nodes = np.where(base_labels == comp_id)[0]
        if len(nodes) <= X:
            new_labels[nodes] = next_cid
            next_cid += 1
            continue

        # Restrict to this component
        unassigned = set(nodes)

        while unassigned:
            # Seed = node with highest remaining degree (tends to form compact chunks)
            def deg_in_remaining(u):
                return sum((v in unassigned) for v in indices[indptr[u]:indptr[u+1]])
            seed = max(unassigned, key=deg_in_remaining)

            # BFS packing up to X nodes
            q = deque([seed])
            unassigned.remove(seed)
            new_labels[seed] = next_cid
            count = 1

            while q and count < X:
                u = q.popleft()
                for v in indices[indptr[u]:indptr[u+1]]:
                    if v in unassigned:
                        unassigned.remove(v)
                        new_labels[v] = next_cid
                        q.append(v)
                        count += 1
                        if count >= X:
                            break
            next_cid += 1

    return new_labels


def haversine_distance_matrix(df, lat_col='lat', lon_col='lon', return_df=False):
    """
    Dense NxN great-circle distance matrix in meters using Haversine.
    """
    coords_deg = df[[lat_col, lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)  # haversine_distances expects radians
    D = haversine_distances(coords_rad) * R  # result in meters
    if return_df:
        return pd.DataFrame(D, index=df.index, columns=df.index)
    return D

def merge_small_clusters_by_min_distance_dense(labels, h_dist_mat, min_size=3, return_mapping=False):
    """
    Merge all small clusters (< min_size) into their nearest large cluster (>= min_size)
    using the precomputed dense haversine distance matrix (in meters).
    Distance between clusters = minimum pairwise distance between any two zones.
    """
    # labels = labels_capped; min_size=3; return_mapping=True
    labels = np.asarray(labels).copy()
    uniq, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(uniq, counts))

    large_ids = [cid for cid, c in size_map.items() if c >= min_size]
    small_ids = [cid for cid, c in size_map.items() if c < min_size]

    if len(small_ids) == 0:
        return (labels, {}) if return_mapping else labels
    if len(large_ids) == 0:
        # fallback: promote largest cluster as the only large cluster
        largest = max(size_map, key=size_map.get)
        large_ids = [largest]
        small_ids = [cid for cid in small_ids if cid != largest]
        if len(small_ids) == 0:
            return (labels, {}) if return_mapping else labels

    mapping = {}
    D = np.asarray(h_dist_mat)
    n = len(D)

    for scid in small_ids:
        # scid = 6
        s_idx = np.where(labels == scid)[0]

        best_dist = np.inf
        best_cid = None

        for lcid in large_ids:
            # lcid = 1
            l_idx = np.where(labels == lcid)[0]
            # compute the minimum pairwise distance between these two clusters
            d_min = np.min(D[np.ix_(s_idx, l_idx)])
            if d_min < best_dist:
                best_dist = d_min
                best_cid = lcid

        if best_cid is not None:
            labels[s_idx] = best_cid
            mapping[scid] = (best_cid, best_dist)

    # reindex labels to 0..K'-1
    _, inv = np.unique(labels, return_inverse=True)

    return (inv, mapping) if return_mapping else inv


def two_stage_capacity_merge(
    labels,
    h_dist_mat,                   # dense NxN, meters
    max_size=50,                  # hard cap per cluster
    min_large_size=3,             # >= this => "large" eligible to receive
    dist_to_large_thresh_m=None,  # max distance small->large; None = no cap
    dist_small_small_thresh_m=None,# max distance small<->small; None = no cap
    order_by="distance_then_size",# or "size_then_distance"
    return_plan=False
):
    """
    Stage 1: assign each small cluster to its nearest large cluster if:
             distance <= dist_to_large_thresh_m (if set) AND capacity fits (<= max_size).
    Stage 2: greedily merge remaining small clusters with each other if:
             distance <= dist_small_small_thresh_m (if set) AND merged size <= max_size.
    Leftovers stay as-is (self clusters). Never exceed max_size.
    """
    labels = np.asarray(labels).copy()
    D = np.asarray(h_dist_mat, dtype=float)
    n = len(labels)

    # --- helpers ---
    def cluster_members_map(labs):
        ids = np.unique(labs)
        return {cid: np.where(labs == cid)[0] for cid in ids}

    def sizes_from_members(mem):
        return {cid: len(idx) for cid, idx in mem.items()}

    def min_pair_dist(idx_a, idx_b):
        return float(np.min(D[np.ix_(idx_a, idx_b)]))

    # initial partitions
    members = cluster_members_map(labels)
    sizes = sizes_from_members(members)
    large_ids = [cid for cid, s in sizes.items() if s >= min_large_size]
    small_ids = [cid for cid, s in sizes.items() if s <  min_large_size]

    if not small_ids:
        return (labels, {}) if return_plan else labels

    # capacity of large clusters
    cap = {cid: max_size - sizes[cid] for cid in large_ids}
    cap = {cid: c for cid, c in cap.items() if c > 0}
    large_ids = [cid for cid in large_ids if cid in cap]

    plan = {}  # record where small clusters go: small_cid -> (target_id, dist_m, stage)

    # -------- Stage 1: small -> nearest large with capacity and distance cap --------
    candidates = []
    for scid in small_ids:
        s_idx = members[scid]
        best = (np.inf, None)
        for lcid in large_ids:
            if cap.get(lcid, 0) <= 0:
                continue
            d = min_pair_dist(s_idx, members[lcid])
            if d < best[0]:
                best = (d, lcid)
        dmin, tgt = best
        candidates.append((scid, dmin, tgt, sizes[scid]))

    if order_by == "size_then_distance":
        candidates.sort(key=lambda t: (-t[3], t[1]))   # big small-clusters first
    else:
        candidates.sort(key=lambda t: (t[1], -t[3]))   # nearest first (default)

    lingering = []
    for scid, dmin, tgt, ssize in candidates:
        ok_dist = (dist_to_large_thresh_m is None) or (dmin <= dist_to_large_thresh_m)
        if tgt is not None and ok_dist and cap.get(tgt, 0) >= ssize:
            # assign
            labels[members[scid]] = tgt
            cap[tgt] -= ssize
            # update members of tgt
            members[tgt] = np.concatenate([members[tgt], members[scid]])
            plan[scid] = (tgt, dmin, "stage1")
            # remove scid from maps
            del members[scid]
            del sizes[scid]
        else:
            lingering.append(scid)

    if not lingering:
        _, inv = np.unique(labels, return_inverse=True)
        return (inv, plan) if return_plan else inv

    # -------- Stage 2: merge small-small under distance & capacity --------
    # We will greedily merge the closest pair that fits the cap, repeatedly.
    # Represent each lingering cluster as a "group".
    groups = {gid: [gid] for gid in lingering}
    group_members = {gid: members[gid].copy() for gid in lingering}
    group_sizes = {gid: sizes[gid] for gid in lingering}
    alive = {gid: True for gid in lingering}

    # Build all candidate pairs (distance, a, b, merged_size), filter by capacity & distance
    pairs = []
    gids = list(groups.keys())
    for i in range(len(gids)):
        for j in range(i+1, len(gids)):
            a, b = gids[i], gids[j]
            merged_sz = group_sizes[a] + group_sizes[b]
            if merged_sz <= max_size:
                d = min_pair_dist(group_members[a], group_members[b])
                if (dist_small_small_thresh_m is None) or (d <= dist_small_small_thresh_m):
                    pairs.append((d, a, b, merged_sz))
    pairs.sort(key=lambda x: x[0])  # nearest first

    # Greedy union
    for d, a, b, merged_sz in pairs:
        if not (alive.get(a, False) and alive.get(b, False)):
            continue
        # still fits?
        if group_sizes[a] + group_sizes[b] <= max_size:
            # merge b -> a
            groups[a].extend(groups[b])
            group_members[a] = np.concatenate([group_members[a], group_members[b]])
            group_sizes[a] += group_sizes[b]
            alive[b] = False

    # Now assign each alive group a *new* cluster id (self clan)
    new_id_start = max(members.keys()) + 1 if members else 0
    for gid, ok in alive.items():
        if not ok:
            continue
        # give this group a new id
        new_cid = new_id_start
        new_id_start += 1
        labels[group_members[gid]] = new_cid
        # record where the original small clusters went (self group)
        for scid in groups[gid]:
            plan[scid] = (new_cid, 0.0, "stage2_self")  # distance not meaningful here

    # -------- reindex labels to 0..K'-1 --------
    _, inv = np.unique(labels, return_inverse=True)
    return (inv, plan) if return_plan else inv

def balltree_proximity_clusters_capped(df, lat_col='lat', lon_col='lon', radius_m=200.0, X=50, min_size = 5, C = 3):
    """
    One-stop helper: build proximity components then apply BFS-based size cap.
    Returns (labels_capped, G, coords_rad).
    """
    base_labels, G, coords_rad = balltree_proximity_clusters(df, lat_col, lon_col, radius_m)
    labels_capped = cap_components_with_bfs(G, base_labels, X=X)
    
    
    h_dist_mat = haversine_distance_matrix(df, 'lat', 'lon')
    
    # new_labels, mapping = merge_small_clusters_by_min_distance_dense(
    #     labels_capped, h_dist_mat, min_size=min_size, return_mapping=True
    # )
    
    # This part is to clean-up those single zones that are not clustered
    # new_labels, plan = two_stage_capacity_merge(
    #     labels=labels_capped,
    #     h_dist_mat=h_dist_mat,
    #     max_size=X,
    #     min_large_size=min_size,
    #     dist_to_large_thresh_m=radius_m,       
    #     dist_small_small_thresh_m= radius_m*C,
    #     return_plan=True
    # )

    return labels_capped, G, h_dist_mat

def pooled_transition_prob(agent_location_staying_only, ignore_state=-1,
                           include_self=True, return_sparse=False):
    """
    Build pooled (across all agents) transition probability matrix P.
    Rows/cols are zone IDs (excluding ignore_state). Transitions are from t -> t+1.

    Behavior:
    - We first compute P including self-transitions.
    - If include_self=False, we set the diagonal to 0 and re-normalize each row to sum to 1.
      Rows that are all zeros remain all zeros.

    If return_sparse=True, returns (CSR matrix, uniques_index).
    Otherwise returns a dense pandas.DataFrame indexed/columned by zone IDs.
    """
    A = agent_location_staying_only.to_numpy()
    O, D = A[:, :-1], A[:,  1:]

    # count transitions for all valid (non-ignore) pairs; self transitions included for now
    mask = (O != ignore_state) & (D != ignore_state)

    if mask.sum() == 0:
        return (csr_matrix((0, 0)), np.array([])) if return_sparse else pd.DataFrame(dtype=float)

    o = O[mask]
    d = D[mask]

    # Map zone ids to [0..K-1] consistently across origins/destinations
    cat, uniques = pd.factorize(np.concatenate([o, d]), sort=True)
    k = len(uniques)
    o_idx = cat[:len(o)]
    d_idx = cat[len(o):]

    # Sparse counts (K x K)
    counts = coo_matrix((np.ones_like(o_idx, dtype=np.int64), (o_idx, d_idx)),
                        shape=(k, k)).tocsr()

    # Row-normalize via diagonal scaling: P = D^{-1} * counts
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums, dtype=float)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    P = diags(inv) @ counts  # CSR

    # If self-transitions should be removed: zero diagonal and re-normalize rows
    if not include_self:
        # zero diagonal
        P = P.tocsr(copy=True)
        P.setdiag(0.0)
        P.eliminate_zeros()
        # re-normalize rows to sum to 1 (leave all-zero rows unchanged)
        rs = np.asarray(P.sum(axis=1)).ravel()
        inv2 = np.zeros_like(rs, dtype=float)
        nz2 = rs > 0
        inv2[nz2] = 1.0 / rs[nz2]
        P = diags(inv2) @ P  # CSR

    if return_sparse:
        return P.tocsr(), uniques

    # Dense DataFrame
    return pd.DataFrame(P.toarray(), index=uniques, columns=uniques)


def build_balltree_adjacency(zones_df, lat_col='lat', lon_col='lon',
                             radius_m=500.0, include_self=True):
    """
    Build a symmetric 0/1 adjacency CSR where edge(i,j)=1 iff Haversine(i,j) <= radius_m.
    """
    coords_deg = zones_df[[lat_col, lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)
    eps_rad = radius_m / R

    tree = BallTree(coords_rad, metric='haversine')
    ind = tree.query_radius(coords_rad, r=eps_rad, return_distance=False)

    rows, cols = [], []
    for i, nbrs in enumerate(ind):
        for j in nbrs:
            rows.append(i); cols.append(j)

    n = len(coords_rad)
    data = np.ones(len(rows), dtype=np.uint8)
    G = csr_matrix((data, (rows, cols)), shape=(n, n)).maximum(
        csr_matrix((data, (cols, rows)), shape=(n, n))
    )
    if include_self:
        G = G + csr_matrix((np.ones(n, dtype=np.uint8), (np.arange(n), np.arange(n))), shape=(n, n))
    G.eliminate_zeros()
    return G 

def overlay_transition_costs_on_adj(zones_df, T_df, G_adj, objective='separate', eps=1e-9):
    """
    Produce a weighted CSR matrix W from adjacency G_adj and transition matrix T_df.
    objective='separate'  -> cost = T  (high T = high cost, encourages cutting)
    objective='keep'      -> cost = 1/(T+eps)  (high T = low cost, encourages keeping)
    """
    # zids = zones_df.index.to_numpy()
    # T = T_df.reindex(index=zids, columns=zids, fill_value=0.0).to_numpy(dtype=float)
    T = T_df.values
    if objective == 'separate':
        cost_mat = T
    elif objective == 'keep':
        cost_mat = 1.0 / (T + eps)
        np.fill_diagonal(cost_mat, 0.0)
    else:
        raise ValueError("objective must be 'separate' or 'keep'")

    # keep costs only where adjacency has edges
    G = G_adj.tocoo()
    mask = G.row < G.col  # undirected unique edges
    r, c = G.row[mask], G.col[mask]
    w = cost_mat[r, c]

    # build symmetric weighted graph
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([w, w])
    W = csr_matrix((data, (rows, cols)), shape=G_adj.shape)
    W.eliminate_zeros()
    return W  # weighted CSR

def mst_skater_from_adj_cost(
    G_adj,                 # CSR (n x n), symmetric 0/1 adjacency (no self-loops)
    W_cost,                # CSR or dense (n x n), non-negative costs (larger = more likely to cut)
    k=None,                # desired number of regions (cut top (k-1) heaviest MST edges globally)
    max_size=None          # optional: enforce ≤ max_size nodes per region (contiguity preserved)
):
    """
    SKATER-style MST partition on a weighted graph:
      - Inputs:
         * G_adj: adjacency that enforces spatial contiguity (0/1, symmetric).
         * W_cost: pairwise costs; only entries present in G_adj are used.
      - Steps:
         1) Mask costs by adjacency -> weighted graph W.
         2) Compute MST(W).
         3) Cut edges by:
            a) max_size (iteratively remove heaviest edges inside oversized components), and/or
            b) k (remove (k-1) heaviest remaining edges globally).
      - Returns:
         labels: np.ndarray of cluster ids (0..K-1)
         G_final: CSR graph after cuts (to inspect)
    """
    # --- shapes & types ---
    if not isinstance(G_adj, csr_matrix):
        G_adj = csr_matrix(G_adj)
    n = G_adj.shape[0]

    # --- build weighted graph W by masking costs with adjacency ---
    if isinstance(W_cost, csr_matrix):
        # use only entries where adjacency has edges
        A = G_adj.tocoo()
        mask = A.row < A.col  # unique undirected edges
        r, c = A.row[mask], A.col[mask]
        w = W_cost[r, c].A1  # extract weights on those pairs
    else:
        # dense array-like
        W_cost = np.asarray(W_cost, dtype=float)
        A = G_adj.tocoo()
        mask = A.row < A.col
        r, c = A.row[mask], A.col[mask]
        w = W_cost[r, c]

    # build symmetric weighted graph W
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([w, w]).astype(float)
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    # ensure symmetry (keep minimal of both directions if duplicates exist)
    W = W.minimum(W.T)
    W.eliminate_zeros()

    # Corner case: no edges
    if W.nnz == 0:
        labels = np.arange(n, dtype=int)
        return labels, csr_matrix((n, n))

    # --- Minimum Spanning Tree on W ---
    mst = minimum_spanning_tree(W).tocsr()
    # undirected MST (mirror weights)
    G = mst.maximum(mst.T)

    # --- helpers ---
    def comp_labels(Gmat):
        _, labs = connected_components(Gmat, directed=False)
        return labs

    def cut_heaviest_edge(Gmat, mask_nodes=None):
        """Remove single heaviest edge in (sub)graph induced by mask_nodes."""
        coo = Gmat.tocoo()
        sel = coo.row < coo.col
        r, c, w = coo.row[sel], coo.col[sel], coo.data[sel]
        if mask_nodes is not None:
            keep = np.isin(r, mask_nodes) & np.isin(c, mask_nodes)
            r, c, w = r[keep], c[keep], w[keep]
        if w.size == 0:
            return False
        j = np.argmax(w)
        Gmat[r[j], c[j]] = 0.0
        Gmat[c[j], r[j]] = 0.0
        Gmat.eliminate_zeros()
        return True

    # --- (1) enforce max_size by iteratively cutting heaviest edges inside oversized components ---
    if max_size is not None:
        while True:
            labs = comp_labels(G)
            sizes = pd.Series(labs).value_counts()
            big = sizes.index[sizes.values > max_size]
            if len(big) == 0:
                break
            for cid in big:
                nodes = np.where(labs == cid)[0]
                # cut one heaviest edge inside this component (repeat next loop if still oversized)
                cut_heaviest_edge(G, mask_nodes=nodes)

    # --- (2) enforce k clusters by cutting heaviest remaining edges globally ---
    if k is not None:
        labs = comp_labels(G)
        num = len(np.unique(labs))
        # if already >= k components (e.g., disconnected adj), stop; else cut to increase to k
        while num < k:
            if not cut_heaviest_edge(G, mask_nodes=None):
                break
            labs = comp_labels(G)
            num = len(np.unique(labs))

    labels = comp_labels(G)
    return labels, G


def apply_indexmap_all(df, indexmap):
    """
    Replace ALL values in df using indexmap, fully vectorized with NumPy.
    - df: DataFrame of integers (zone ids).
    - indexmap: pd.Series or dict mapping zone_index -> cluster_id.
    Returns: DataFrame with same shape/index/columns.
    """
    if isinstance(indexmap, dict):
        indexmap = pd.Series(indexmap)

    # ensure -1 -> -1
    if -1 not in indexmap.index:
        indexmap.loc[-1] = -1

    # Build mapping array (needs non-negative indices)
    min_idx = indexmap.index.min()
    max_idx = indexmap.index.max()
    offset = -min_idx if min_idx < 0 else 0

    lut = np.arange(min_idx, max_idx+1)  # identity mapping by default
    lut = lut.astype(indexmap.dtype)

    # override with indexmap values
    for k, v in indexmap.items():
        lut[k + offset] = v

    arr = df.to_numpy()
    mapped = lut[arr + offset]

    return pd.DataFrame(mapped, index=df.index, columns=df.columns)