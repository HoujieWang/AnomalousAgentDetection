# In[1]:
''' This file reads the raw knoxville data, and perform the following analysis:
    1. For raw lat/lon coordinates, if at 5-min level, the H-dist shift is lower than 50m, 
        we regard that as fake transition and force the new lat/lon to be the previous one
        
    2. Discretize the fixed lat/lon in Step 1 into space cells with 500m side size
    
    3. Run spatial clustering, based on BallTree constructed adjacency matrix where 
        lat/lon are below certain threshold. Use BFS to refine larger clusters to meet max number cmponent constraint
        
    This file runs clustering on passing and staying zones separately. Note that now we do balltree clustering, 
    and then merge or recluster the small clusters such that we do not have too many very small clusters lingering around.
'''

import Utilities2025.geo_utilities as mygeo
import Utilities2025.clustering_utilities as mycl
import Utilities2025.counting_utilities as mycount
import pandas as pd 
import numpy as np
import os
import pyarrow
import time
import pickle
import matplotlib as mpl
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


from typing import Dict, Tuple,  Optional, Union
from pybats.dglm import bin_dglm
from pathlib import Path

data_directory = '/Users/wanghoujie/Downloads/DynamicNetwork/all_agents/str_data/train'


# In[2]:


''' Initialize an empty dictionary to store the data '''
data_dict = {}

global_bounds = {'lat_min': 35.51009030444566, \
                 'lat_max': 36.5818509, \
                 'lon_min': -84.95347326088375, \
                 'lon_max': -83.70374631780497}
# This is the side length of each cell in lat/lon unit, which is around CELL/0.001 * 111 meters

minimum_movement_dist = 50
CELL = 0.01

# Earth radius (m)
R = 6371008.8 

lat_min = global_bounds['lat_min']
lon_min = global_bounds['lon_min']
n_rows = int(np.ceil((global_bounds['lat_max'] - lat_min) / CELL))
n_cols = int(np.ceil((global_bounds['lon_max'] - lon_min) / CELL))

# Distance for two zones to be considered adjacent
radius_m  = 1250
# maximum number of zones per cluster
max_size = 15
# minimum number of zones per cluster
min_size = 3


# In[7]:


''' Code to summarize the cell into a wide format, each row denotes an agent trajectory '''
timestamp = None
for x in range(50):
    print(x)
    # List all files in the directory that start with the current prefix
    prefix = f"agent=subpop_{x}_agent_"
    files = [f for f in os.listdir(data_directory) if f.startswith(prefix)]
    
    # Reorder the files
    files = [files[f] for f in np.argsort([int(file.split('_')[-1]) for file in files])]
    
    # Load each file and store it in the dictionary with key (x, y)
    for file in files:
        # Extract y from the filename
        y = int(file.split('_')[-1])
        file_path = os.path.join(data_directory, file)
#         data_dict[(x, y)] = pd.read_parquet(file_path)
#         if timestamp is None:
#             timestamp = data_dict[(x, y)]['timestamp'].dt.floor("5min")
        
#         data_dict[(x, y)]['subpop'] = x
#         data_dict[(x, y)]['agent_id'] = y
        
#         data_dict[(x, y)] = data_dict[(x, y)].drop(columns = ['timestamp', 'time_bin', 'geocode'])
        
#         data_dict[(x, y)]['zone_index'] = mygeo.latlon_to_index(data_dict[(x, y)]['latitude'].values,
#                                                                 data_dict[(x, y)]['longitude'].values,
#                                                                 lat_min, lon_min, n_rows, n_cols, CELL)
        tmp = pd.read_parquet(file_path)
        if timestamp is None:
                    timestamp = tmp['timestamp'].dt.floor("5min")
        tmp['subpop'] = x
        tmp['agent_id'] = y

        df_adj = mygeo.adjust_trajectory_single(tmp, thresh_m=minimum_movement_dist)
        df_final = mygeo.hier_cluster_haversine(df_adj[['lat_adj', 'lon_adj']].drop_duplicates(), \
                                          lat_col='lat_adj', lon_col='lon_adj', t_m=minimum_movement_dist, method='average')


        mapping = (
            df_final
            .set_index(['lat_adj', 'lon_adj'])['cluster_id']
            .to_dict()
        )

        df_adj['cluster_id'] = df_adj.set_index(['lat_adj', 'lon_adj']).index.map(mapping)


        df_adj['lat_adj'] = df_adj.groupby('cluster_id')['lat_adj'].transform('mean')
        df_adj['lon_adj'] = df_adj.groupby('cluster_id')['lon_adj'].transform('mean')

        df_adj['zone_index'] = mygeo.latlon_to_index(df_adj['lat_adj'].values,
                                                    df_adj['lon_adj'].values,
                                                    lat_min, lon_min, n_rows, n_cols, CELL)

        df_adj = df_adj.\
            rename(columns = {'latitude': 'lat_raw', 'longitude': 'lon_raw',
                                'lat_adj': 'latitude', 'lon_adj': 'longitude'}).\
            drop(columns = ['timestamp', 'time_bin', 'geocode', 'step_m'])
        data_dict[(x, y)] = df_adj


# #### Combines all the data and then obtain the zone index matrix
all_agent_df = pd.concat(list(data_dict.values()), axis = 0)
agent_location = np.array([x['zone_index'] for x in data_dict.values()])
agent_location_df = pd.DataFrame(agent_location, \
              index = list(data_dict.keys()),\
              columns=timestamp.dt.strftime("%Y-%m-%d-%H:%M"))
agent_location_df.columns.name = None
agent_location_df.to_parquet(f'agent_location_str_{int((CELL / 0.001)*100)}m_agg.parquet')


# In[3]:


''' Load the saved trajectory file '''

agent_location = pd.read_parquet(f'agent_location_str_{int((CELL / 0.001)*100)}m_agg.parquet')
agent_location.index = pd.Index([(x[0], x[1]) for x in agent_location.index])

#### Visualize the spatial distribution of the geo-locations
zones_visited = np.unique(agent_location.values)


# In[4]:


''' Get the staying and passing zones for each agent, we define so by based on the averaged staying times'''
t0 = time.perf_counter()
agent_zone_staying_times = mygeo.compute_stay_lengths_by_agent(agent_location)
staying_zones_per_agent = {key: pd.Series({key2: np.mean(val2) \
                                           for key2, val2 in val.items() if np.mean(val2) > 1 and np.sum(val2 > 1) >=2 }) \
 for key, val in agent_zone_staying_times.items()}

passing_zones_per_agent = {key: pd.Series({key2: np.mean(val2) \
                                           for key2, val2 in val.items() if not (np.mean(val2) > 1 and np.sum(val2 > 1) >=2)}).index.tolist() \
 for key, val in agent_zone_staying_times.items()}  
t1 = time.perf_counter()
print(f"Total time: {t1 - t0:.2f} s")


# In[ ]:


#### Print out the distribution of number of staying zones by duration bucket
levels = {
    ">=8h": lambda v: v >= 96,                      # >= 8h
    "4h-8h": lambda v: (v >= 48) & (v < 96),        # 4h–8h
    "2h-4h": lambda v: (v >= 24) & (v < 48),        # 2h–4h
    "0-2h":  lambda v: (v >= 0) & (v < 24),         # <2h
}

# The table represents there are x number of agents who have row number of column name of cell
results = {}
total_levels = len(levels)

for i_level, (label, cond) in enumerate(levels.items(), 1):
    counts_dict = {}
    total_agents = len(staying_zones_per_agent)

    for i_agent, (key, val) in enumerate(staying_zones_per_agent.items(), 1):
        counts_dict[key] = len(val[cond(val)])

        if i_agent % 1000 == 0 or i_agent == total_agents:
            # clear_output(wait=True)
            print(f"[{label}] Processed {i_agent}/{total_agents} agents")

    results[label] = pd.Series(counts_dict).value_counts().sort_index()

    print(f"Finished {i_level}/{total_levels} levels")

df_results = pd.DataFrame(results).fillna(0).astype(int)
print(df_results)


# In[ ]:


bucket_zones = {label: set() for label in levels}

total_agents = len(staying_zones_per_agent)

for agent_idx, (agent_key, agent_dict) in enumerate(staying_zones_per_agent.items(), 1):
    total_zones = len(agent_dict)
    for zone_idx, (zone_id, lens) in enumerate(agent_dict.items(), 1):
        a = np.atleast_1d(lens)
        if a.size == 0:
            continue
        a = a[~np.isnan(a)]
        if a.size == 0:
            continue
        a = a.astype(int, copy=False)
        for label, cond in levels.items():
            if cond(a).any():
                bucket_zones[label].add(int(zone_id))


    # # print progress every 1000 agents
    # if agent_idx % 1000 == 0 or agent_idx == total_agents:
    #     # clear_output(wait=True)
    #     print(f"Processed agent {agent_idx}/{total_agents}")


# In[8]:


# Optional: convert sets to sorted lists
bucket_zones = {k: sorted(v) for k, v in bucket_zones.items()}

# Summary counts
summary = pd.Series({k: len(v) for k, v in bucket_zones.items()}).sort_index()
print(summary)


# In[3]:


to_save = {
    "agent_location": agent_location, 
    "zones_visited": zones_visited, 
    "agent_zone_staying_times": agent_zone_staying_times,
    "staying_zones_per_agent": staying_zones_per_agent,
    "passing_zones_per_agent": passing_zones_per_agent,  
    "bucket_zones": bucket_zones
}
with open(f"key_var_upto_bucket_zones_{int((CELL / 0.001)*100)}m.pkl", "wb") as f:
    pickle.dump(to_save, f)

with open(f"key_var_upto_bucket_zones_{int((CELL / 0.001)*100)}m.pkl", "rb") as f:
    data = pickle.load(f)


# In[4]:
bucket_zones = data['bucket_zones']
zones_visited = data['zones_visited']
agent_location = data['agent_location']
staying_zones_per_agent = data['staying_zones_per_agent']
passing_zones_per_agent = data['passing_zones_per_agent']


# In[ ]:


# -------- Prepare staying zones data --------
# out = mygeo.zone_id_to_rowcol_latlon(
#     np.array(bucket_zones['>=8h']), n_rows, n_cols, lat_min, lon_min, CELL
# )
# -------- Combine all zones across all buckets --------
all_staying_zones = np.unique(
    np.concatenate([np.array(v, dtype=int) for v in bucket_zones.values()])
)
out = mygeo.zone_id_to_rowcol_latlon(
    all_staying_zones, n_rows, n_cols, lat_min, lon_min, CELL
)

zone_loc_stay = pd.DataFrame(out[2:]).T
zone_loc_stay.columns = ['lat', 'lon']

coords_deg = np.c_[zone_loc_stay['lat'], zone_loc_stay['lon']]
coords_rad = np.radians(coords_deg)

eps_km = 2
eps_rad = eps_km / 6371.0088

db = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine').fit(coords_rad)
labels = db.labels_
zone_loc_stay['cluster'] = labels
zone_loc_stay['zone_index'] = mygeo.latlon_to_index(zone_loc_stay['lat'].values,
                      zone_loc_stay['lon'].values,
                      lat_min, lon_min, n_rows, n_cols, CELL)

# -------- Prepare passing zones data --------
passing_zones = zones_visited[~np.isin(zones_visited, all_staying_zones)]
passing_out = mygeo.zone_id_to_rowcol_latlon(
    passing_zones, n_rows, n_cols, lat_min, lon_min, CELL
)
passing_loc = pd.DataFrame(passing_out[2:]).T
passing_loc.columns = ['lat', 'lon']
passing_loc['zone_index'] = mygeo.latlon_to_index(passing_loc['lat'].values,
                      passing_loc['lon'].values,
                      lat_min, lon_min, n_rows, n_cols, CELL)

# In[ ]:


# -------- Plot: 2 rows x 2 columns --------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- Row 1 / Col 1: Raw staying + passing ---
axes[0, 0].scatter(passing_loc['lon'], passing_loc['lat'],
                   s=0.05, color='lightgray', alpha=0.5, label='Passing Zones')
axes[0, 0].scatter(zone_loc_stay['lon'], zone_loc_stay['lat'],
                   s=0.1, color='black', alpha=0.7, label='Staying Zones')
axes[0, 0].set_title("Raw Staying Zones + Passing Zones")
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")
axes[0, 0].set_aspect('equal', adjustable='box')
axes[0, 0].grid(True, linewidth=0.3, alpha=0.4)
axes[0, 0].legend(markerscale=20, fontsize=8)

# --- Row 1 / Col 2: DBSCAN clusters + passing ---
axes[0, 1].scatter(passing_loc['lon'], passing_loc['lat'],
                   s=0.05, color='lightgray', alpha=1, label='Passing Zones')
scatter = axes[0, 1].scatter(zone_loc_stay['lon'], zone_loc_stay['lat'],
                             c=zone_loc_stay['cluster'], cmap='tab20',
                             s=0.5, alpha=0.7, label='Staying Zones')
axes[0, 1].set_title(f"DBSCAN Clusters ({int(eps_km)} km eps) + Passing Zones")
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")
axes[0, 1].set_aspect('equal', adjustable='box')
axes[0, 1].grid(True, linewidth=0.3, alpha=0.4)

for cluster_id in np.unique(labels):
    cluster_points = zone_loc_stay[zone_loc_stay['cluster'] == cluster_id]
    if len(cluster_points) == 0:
        continue
    center_lon = cluster_points['lon'].mean()
    center_lat = cluster_points['lat'].mean()
    axes[0, 1].text(center_lon, center_lat, str(cluster_id), fontsize=8,
                    ha='center', va='center', weight='bold')

fig.colorbar(scatter, ax=axes[0, 1], label="Cluster ID", fraction=0.046, pad=0.04)

# --- Row 2 / Col 1: Passing zones only ---
axes[1, 0].scatter(passing_loc['lon'], passing_loc['lat'],
                   s=0.2, color='lightgray', alpha=0.7, label='Passing Zones')
axes[1, 0].set_title("Passing Zones Only")
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")
axes[1, 0].set_aspect('equal', adjustable='box')
axes[1, 0].grid(True, linewidth=0.3, alpha=0.4)
axes[1, 0].legend(markerscale=20, fontsize=8)

# --- Row 2 / Col 2: Staying zones only ---
axes[1, 1].scatter(zone_loc_stay['lon'], zone_loc_stay['lat'],
                   s=0.5, color='black', alpha=0.7, label='Staying Zones')
axes[1, 1].set_title("Staying Zones Only")
axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")
axes[1, 1].set_aspect('equal', adjustable='box')
axes[1, 1].grid(True, linewidth=0.3, alpha=0.4)
axes[1, 1].legend(markerscale=20, fontsize=8)

plt.show()
    
# In[11]:

#### Clustering passing zones based on proximity and capped on cluster size
labels_passing_raw, G, h_dist = mycl.balltree_proximity_clusters_capped(passing_loc, 'lat', 'lon',
                                                  radius_m=radius_m, X=max_size, min_size = min_size)

labels_pssing, mapping = mycl.merge_small_clusters_by_min_distance_dense(
    labels_passing_raw, h_dist, min_size=min_size, return_mapping=True
)

passing_loc['cluster_id'] = -(labels_pssing+1)

# In[23]:

#### Plotting the clusters on the map
def plot_partitions_with_cycle_per_panel(passing_loc, cluster_col='cluster_id',
                                         n_partitions=10, ncols=2,
                                         max_legend_items=20):
    """
    Use the same palette (Matplotlib default cycle) across all partitions.
    Within each partition, assign different colors to its clusters by cycling the palette.
    """
    clusters = np.sort(passing_loc[cluster_col].unique())
    parts = np.array_split(clusters, n_partitions)
    nrows = int(np.ceil(n_partitions / ncols))

    # default color cycle
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    C = len(cycle)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), constrained_layout=True)
    axes = np.array(axes).ravel()

    bg_lon = passing_loc['lon'].to_numpy()
    bg_lat = passing_loc['lat'].to_numpy()
    labels_arr = passing_loc[cluster_col].to_numpy()

    for pidx, part in enumerate(parts):
        ax = axes[pidx]
        # background: all points in light grey
        ax.scatter(bg_lon, bg_lat, s=2, color='lightgrey', alpha=0.45)

        # assign colors per partition: different color for each cluster in this part
        for i, cid in enumerate(part):
            color = cycle[i % C]  # reuse the same palette across partitions
            mask = (labels_arr == cid)
            ax.scatter(
                passing_loc.loc[mask, 'lon'],
                passing_loc.loc[mask, 'lat'],
                s=4, color=color, alpha=0.9, label=str(cid)
            )

        if len(part) <= max_legend_items:
            ax.legend(title="cluster_id", fontsize=8, markerscale=2, frameon=True)

        ax.set_title(f'Partition {pidx+1}/{n_partitions} (clusters: {len(part)})')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # hide unused axes if any
    for j in range(pidx + 1, len(axes)):
        axes[j].axis('off')

    plt.show()


plot_partitions_with_cycle_per_panel(passing_loc, cluster_col='cluster_id',
                                      n_partitions=15, ncols=2)

# In[27]:



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


# In[28]:

#### Get transition probability matrix of among staying zones

agent_location_staying_only = agent_location.copy()
agent_location_staying_only[~agent_location_staying_only.isin(all_staying_zones)] = -1

# %%

#### Check how much transition is captured after clustering

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

# Build mapping from zone_loc_stay
# indexmap = zone_loc_stay[['zone_index','cluster_id']].set_index('zone_index')['cluster_id']
# indexmap.loc[-1] = -1
# agent_location_staying_mapped = mycl.apply_indexmap_all(agent_location_staying_only, indexmap=indexmap)


# labels_stay_nocost, _, _ = mycl.balltree_proximity_clusters_capped(zone_loc_stay, 'lat', 'lon',
#                                                   radius_m=radius_m, X=max_size, min_size = min_size)


base_labels, G, coords_rad = mycl.balltree_proximity_clusters(zone_loc_stay, 'lat', 'lon', radius_m)
labels_capped = mycl.cap_components_with_bfs(G, base_labels, X=max_size)

h_dist_mat = mycl.haversine_distance_matrix(zone_loc_stay, 'lat', 'lon')
new_labels, plan = mycl.two_stage_capacity_merge(
    labels=labels_capped,
    h_dist_mat=h_dist_mat,
    max_size=max_size,
    min_large_size=min_size,
    dist_to_large_thresh_m=radius_m,       
    dist_small_small_thresh_m=radius_m*3,
    order_by='size_then_distance',
    return_plan=True
)


# zone_loc_stay['cluster_id_nocost'] = labels_stay_nocost
zone_loc_stay['cluster_id'] = new_labels


indexmap_nocost = zone_loc_stay[['zone_index','cluster_id']].set_index('zone_index')['cluster_id']
indexmap_nocost.loc[-1] = -1
agent_location_staying_mapped_nocost = mycl.apply_indexmap_all(agent_location_staying_only, indexmap=indexmap_nocost)

# In[266]:

def plot_partitions_with_zoom_pair(
    passing_loc, cluster_col='cluster_id',
    n_partitions=10, zoom_factor=1.25,
    max_legend_items=20, min_pad_deg=1e-4,
    save_row_i=None,            # <- which row to save (1-based). Use -1 for last, -2 for second-last, etc.
    save_dir="figure_results",
    save_name=None              # if None, will auto-name like "passing_clustering_row{i}.pdf"
):
    """
    Plots an n_partitions x 2 grid (full map + zoom for each partition).
    Additionally, if save_row_i is set, re-draws ONLY that row into a fresh figure and saves as PDF.

    save_row_i:
      - Positive k => save the k-th row (1-based).
      - Negative -k => save the k-th row from the end (-1 last, -2 second-last, ...).
      - None => do not save a single-row figure.
    """
    # ---- prep ----
    clusters = np.sort(passing_loc[cluster_col].unique())
    parts = np.array_split(clusters, n_partitions)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    C = len(cycle)

    bg_lon = passing_loc['lon'].to_numpy()
    bg_lat = passing_loc['lat'].to_numpy()
    labels_arr = passing_loc[cluster_col].to_numpy()

    fig, axes = plt.subplots(n_partitions, 2, figsize=(12, 5*n_partitions), constrained_layout=True)
    if n_partitions == 1:
        axes = np.array([axes])  # ensure 2D indexing

    # normalize save_row_i -> 0-based index
    idx_to_save = None
    if save_row_i is not None:
        if save_row_i == 0:
            raise ValueError("save_row_i is 1-based (use 1..n or negative indices like -1).")
        idx_to_save = (save_row_i - 1) if save_row_i > 0 else (n_partitions + save_row_i)
        if not (0 <= idx_to_save < n_partitions):
            raise ValueError(f"save_row_i resolves to index {idx_to_save}, out of range 0..{n_partitions-1}.")

    selected_points = []   # (lon_i, lat_i, color) for the row to save
    selected_part = None
    selected_zoom_limits = None

    # ---- draw grid ----
    for pidx, part in enumerate(parts):
        ax_full = axes[pidx, 0]
        ax_zoom = axes[pidx, 1]

        # base background
        ax_full.scatter(bg_lon, bg_lat, s=2, color='lightgrey', alpha=0.45)
        ax_zoom.scatter(bg_lon, bg_lat, s=2, color='lightgrey', alpha=0.45)

        highlighted_any = False
        part_lons, part_lats = [], []

        # plot each cluster in this partition (same colors on both subplots)
        for i, cid in enumerate(part):
            color = cycle[i % C]
            mask = (labels_arr == cid)
            if not np.any(mask):
                continue
            highlighted_any = True
            lon_i = passing_loc.loc[mask, 'lon'].to_numpy()
            lat_i = passing_loc.loc[mask, 'lat'].to_numpy()
            part_lons.append(lon_i)
            part_lats.append(lat_i)

            ax_full.scatter(lon_i, lat_i, s=4, color=color, alpha=0.9, label=str(cid))
            ax_zoom.scatter(lon_i, lat_i, s=4, color=color, alpha=0.9)

            # if this row is the one to save, remember points/colors
            if idx_to_save is not None and pidx == idx_to_save:
                selected_points.append((lon_i, lat_i, color))

        if highlighted_any and (len(part) <= max_legend_items):
            ax_full.legend(title="cluster_id", fontsize=8, markerscale=2, frameon=True)

        ax_full.set_title(f'Partition {pidx+1}/{n_partitions} — Full view (clusters: {len(part)})')
        ax_full.set_aspect('equal', adjustable='box')
        ax_full.grid(True, linewidth=0.3, alpha=0.4)
        ax_full.set_xlabel("Longitude"); ax_full.set_ylabel("Latitude")

        ax_zoom.set_title('Zoomed view')
        ax_zoom.set_aspect('equal', adjustable='box')
        ax_zoom.grid(True, linewidth=0.3, alpha=0.4)
        ax_zoom.set_xlabel("Longitude"); ax_zoom.set_ylabel("Latitude")

        # compute zoom limits for this partition
        if highlighted_any:
            lons_part = np.concatenate(part_lons)
            lats_part = np.concatenate(part_lats)

            lon_min, lon_max = float(lons_part.min()), float(lons_part.max())
            lat_min, lat_max = float(lats_part.min()), float(lats_part.max())
            lon_rng = max(lon_max - lon_min, 2*min_pad_deg)
            lat_rng = max(lat_max - lat_min, 2*min_pad_deg)
            lon_mid = (lon_min + lon_max) / 2.0
            lat_mid = (lat_min + lat_max) / 2.0

            lon_half = (zoom_factor * lon_rng) / 2.0
            lat_half = (zoom_factor * lat_rng) / 2.0

            xlim = (lon_mid - lon_half, lon_mid + lon_half)
            ylim = (lat_mid - lat_half, lat_mid + lat_half)
            ax_zoom.set_xlim(*xlim); ax_zoom.set_ylim(*ylim)

            if idx_to_save is not None and pidx == idx_to_save:
                selected_zoom_limits = (xlim, ylim)
                selected_part = part

    plt.show()

    # ---- re-draw and save ONLY the chosen row into a fresh figure ----
    if idx_to_save is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if save_name is None:
            save_name = f"passing_clustering_row{idx_to_save+1}.pdf"
        save_path = str(Path(save_dir) / save_name)

        fig_last, axes_last = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # Left: full view
        axL = axes_last[0]
        axL.scatter(bg_lon, bg_lat, s=2, color='lightgrey', alpha=0.45)
        for lon_i, lat_i, color in selected_points:
            axL.scatter(lon_i, lat_i, s=4, color=color, alpha=0.9)
        num_clusters = len(selected_part) if selected_part is not None else 0
        axL.set_title(f'Partition {idx_to_save+1}/{n_partitions} — Full view (clusters: {num_clusters})')
        axL.set_aspect('equal', adjustable='box')
        axL.grid(True, linewidth=0.3, alpha=0.4)
        axL.set_xlabel("Longitude"); axL.set_ylabel("Latitude")

        # Right: zoomed view
        axR = axes_last[1]
        axR.scatter(bg_lon, bg_lat, s=2, color='lightgrey', alpha=0.45)
        for lon_i, lat_i, color in selected_points:
            axR.scatter(lon_i, lat_i, s=4, color=color, alpha=0.9)
        if selected_zoom_limits is not None:
            (x0, x1), (y0, y1) = selected_zoom_limits
            axR.set_xlim(x0, x1); axR.set_ylim(y0, y1)
        axR.set_title('Zoomed view')
        axR.set_aspect('equal', adjustable='box')
        axR.grid(True, linewidth=0.3, alpha=0.4)
        axR.set_xlabel("Longitude"); axR.set_ylabel("Latitude")

        fig_last.savefig(save_path, format="pdf")
        plt.close(fig_last)
        print(f"Saved row {idx_to_save+1} to: {save_path}")
        
        
        
plot_partitions_with_zoom_pair(zone_loc_stay, cluster_col='cluster_id',
                               n_partitions=20, zoom_factor=3, save_row_i = -2, \
                                   save_name='staying_example.pdf')

plot_partitions_with_zoom_pair(passing_loc, cluster_col='cluster_id',
                               n_partitions=15, zoom_factor=3, save_last=False)

plot_partitions_with_zoom_pair(zone_loc_stay, cluster_col='cluster_id',
                               n_partitions=15, zoom_factor=3)


# In[267]:
''' Once we obtain the agent location in clusters,  we get transition-wise time series '''
    
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

#### Testing example
# test_df = pd.DataFrame({
#     "t0": [1, 2, 1],   # agent0 in 1, agent1 in 2, agent2 in 1
#     "t1": [2, 2, 3],
#     "t2": [3, 3, 3],
#     "t3": [3, 1, 3]
# }, index=["agent0", "agent1", "agent2"])
    
# M_tst, row_idx_tst, time_cols_tst = timewise_transition_counts_sparse_streaming(
#     test_df, lag=1, include_self=True
# )
# M_tst, row_idx_tst = reorder_rows_within_origin_by_totals(M_tst, row_idx_tst,
#                                                 origin_asc=True,
#                                                 totals_desc=True)

# pd.DataFrame.sparse.from_spmatrix(M_tst, index=row_idx_tst, columns=time_cols_tst)
    
agent_location_cl = mycl.apply_indexmap_all(agent_location, \
                        indexmap=pd.concat([zone_loc_stay[['zone_index','cluster_id']], \
                                            passing_loc[['zone_index','cluster_id']]], axis=0).set_index('zone_index')['cluster_id'])

agent_location_cl.to_parquet(f"agent_location_cl_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.parquet", engine="pyarrow", index=True)

 
# In[268]:

lat_lon_map = pd.concat([zone_loc_stay[['zone_index', 'lat', 'lon']], \
                         passing_loc[['zone_index', 'lat', 'lon']]], axis=0).set_index('zone_index')

traj = lat_lon_map.loc[agent_location.iloc[0,:]]
traj['time'] = pd.to_datetime(agent_location.columns)

plt.close('all')
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

# -----------------------------
# Background zones
# -----------------------------
ax.scatter(
    passing_loc['lon'], passing_loc['lat'],
    s=0.03, color='gray', alpha=1, label='Global passing zones'
)
ax.scatter(
    zone_loc_stay['lon'], zone_loc_stay['lat'],
    s=0.2, color='black', alpha=1, label='Global staying Zones'
)

# -----------------------------
# Trajectory preparation
# -----------------------------
traj_sorted = traj.sort_values("time").copy()

# --- compute consecutive dwell run length ---
# a new run starts when zone_index changes
zone_id = traj_sorted.index.to_numpy()
new_run = np.concatenate([[True], zone_id[1:] != zone_id[:-1]])
run_id = np.cumsum(new_run)

# run length for each run
run_lengths = (
    traj_sorted
    .assign(run_id=run_id)
    .groupby("run_id")
    .size()
)

traj_sorted["run_len"] = [run_lengths[r] for r in run_id]

# staying zone membership by coordinate
stay_coords = set(zip(zone_loc_stay["lat"].round(8), zone_loc_stay["lon"].round(8)))
traj_sorted["is_stay"] = [
    (round(la, 8), round(lo, 8)) in stay_coords
    for la, lo in zip(traj_sorted["lat"], traj_sorted["lon"])
]

# -----------------------------
# Color by run length (optional but nice)
# -----------------------------
# dwell = np.log(traj_sorted["run_len"].to_numpy(dtype=float))
# cap = np.quantile(dwell, 0.95) if len(dwell) > 10 else dwell.max()
# cap = max(cap, 1.0)
# dwell_norm = np.clip(dwell / cap, 0.0, 1.0)

dwell = traj_sorted["run_len"].to_numpy(dtype=float)
dwell_log = np.log1p(dwell)
cap = np.quantile(dwell_log, 0.95)
dwell_norm = np.clip(dwell_log / cap, 0.0, 1.0)

cmap = plt.get_cmap("plasma")
colors = cmap(dwell_norm)

alpha_stay, alpha_pass = 0.9, 0.25
alphas = np.where(np.array(traj_sorted["is_stay"]), alpha_stay, alpha_pass)
colors[:, 3] = alphas

# -----------------------------
# Point size:
# small by default, slightly larger if run_len >= 6
# -----------------------------
base_size = 6
big_size = 18   # for >=30min dwell
sizes = np.where(traj_sorted["run_len"] >= 6, big_size, base_size)

# -----------------------------
# Plot trajectory
# -----------------------------
ax.plot(
    traj_sorted["lon"].to_numpy(),
    traj_sorted["lat"].to_numpy(),
    linewidth=0.5,
    alpha=0.25,
    zorder=4
)

ax.scatter(
    traj_sorted["lon"].to_numpy(),
    traj_sorted["lat"].to_numpy(),
    s=sizes,
    c=colors,
    linewidths=0,
    label="Agent trajectory (larger = ≥30min dwell)",
    zorder=5
)

# -----------------------------
# Formatting
# -----------------------------
ax.set_title("Agent 1's Trajectory on Day 1")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linewidth=0.3, alpha=0.4)
ax.legend(markerscale=4, fontsize=8, loc="best")

norm = mpl.colors.Normalize(vmin=0, vmax=cap)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)

# --- original-scale ticks ---
tick_vals = np.array([1, 3, 6, 12, 24, 48, 96])
tick_pos = np.clip(np.log1p(tick_vals), 0, cap)

cbar.set_ticks(tick_pos)
cbar.set_ticklabels([str(v) for v in tick_vals])
cbar.set_label("Consecutive dwell length (5-min bins)")
fig.tight_layout()
plt.show()

fig.savefig(
    "figure_results/agent_trajectory_overlay.pdf",
    format="pdf",
    bbox_inches="tight"
)








