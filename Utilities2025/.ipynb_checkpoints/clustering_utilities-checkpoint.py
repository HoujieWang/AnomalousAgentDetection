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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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


def balltree_proximity_clusters_capped(df, lat_col='lat', lon_col='lon', radius_m=200.0, X=50):
    """
    One-stop helper: build proximity components then apply BFS-based size cap.
    Returns (labels_capped, G, coords_rad).
    """
    base_labels, G, coords_rad = balltree_proximity_clusters(df, lat_col, lon_col, radius_m)
    labels_capped = cap_components_with_bfs(G, base_labels, X=X)
    return labels_capped, G, coords_rad