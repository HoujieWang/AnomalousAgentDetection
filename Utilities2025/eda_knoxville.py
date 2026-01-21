import os
os.chdir("/Users/wanghoujie/Documents/GitHub/BDFM_Python/Utilities")
import scipy.linalg
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.sparse import lil_array
import time
import datetime
import statsmodels.api as sm
from scipy.special import logit, expit, loggamma, polygamma
from scipy.stats import norm, shapiro, beta
from numpy import exp, log, quantile, log10
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import xticks
import copy
from datetime import datetime
from Poisson import *
from Bernoulli import *
from flow_counter import *
import scipy.sparse
from scipy.sparse import csr_matrix, eye, csr
import pyarrow
import pickle
import datetime
import sys
import warnings
import math
import scipy.sparse as sps
import numpy.matlib
import scipy.io
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import DBSCAN

# from pyspark import SparkContext, SparkConf
# from pyspark.sql import SparkSession

os.chdir("/Users/wanghoujie/Downloads/DynamicNetwork")
import Utilities2025.geo_utilities as mygeo
directory_path = 'all_agents/str_data/train'

# Initialize an empty dictionary to store the data
data_dict = {}

global_bounds = {'lat_min': 35.51009030444566, \
                 'lat_max': 36.5818509, \
                 'lon_min': -84.95347326088375, \
                 'lon_max': -83.70374631780497}
CELL = 0.001
lat_min = global_bounds['lat_min']
lon_min = global_bounds['lon_min']
n_rows = int(np.ceil((global_bounds['lat_max'] - lat_min) / CELL))
n_cols = int(np.ceil((global_bounds['lon_max'] - lon_min) / CELL))


# timestamp = None
# for x in range(50):
#     print(x)
#     # List all files in the directory that start with the current prefix
#     prefix = f"agent=subpop_{x}_agent_"
#     files = [f for f in os.listdir(directory_path) if f.startswith(prefix)]
    
#     # Reorder the files
#     files = [files[f] for f in np.argsort([int(file.split('_')[-1]) for file in files])]
    
#     # Load each file and store it in the dictionary with key (x, y)
#     for file in files:
#         # Extract y from the filename
#         y = int(file.split('_')[-1])
#         file_path = os.path.join(directory_path, file)
#         data_dict[(x, y)] = pd.read_parquet(file_path)
#         if timestamp is None:
#             timestamp = data_dict[(x, y)]['timestamp'].dt.floor("5min")
        
#         data_dict[(x, y)]['subpop'] = x
#         data_dict[(x, y)]['agent_id'] = y
        
#         data_dict[(x, y)] = data_dict[(x, y)].drop(columns = ['timestamp', 'time_bin', 'geocode'])
        
#         data_dict[(x, y)]['zone_index'] = mygeo.latlon_to_index(data_dict[(x, y)]['latitude'].values,
#                                                                 data_dict[(x, y)]['longitude'].values,
#                                                                 lat_min, lon_min, n_rows, n_cols, CELL)

# #### Combines all the data and then obtain the zone index matrix
# all_agent_df = pd.concat(list(data_dict.values()), axis = 0)
# agent_location = np.array([x['zone_index'] for x in data_dict.values()])
# agent_location_df = pd.DataFrame(agent_location, \
#               index = list(data_dict.keys()),\
#               columns=timestamp.dt.strftime("%Y-%m-%d-%H:%M"))
# agent_location_df.columns.name = None
# agent_location_df.to_parquet('all_agents/str_data/agent_location_str_100m.parquet')

# mygeo.zone_id_to_rowcol_latlon(agent_location_df.iloc[0,:],
#                                n_rows, n_cols, lat_min, lon_min, CELL)


# del all_agent_df
# agent_location = agent_location_df
# del agent_location_df
# del data_dict


agent_location = pd.read_parquet('all_agents/str_data/agent_location_str_100m.parquet')
agent_location.index = pd.Index([(x[0], x[1]) for x in agent_location.index])
#### Visualize the spatial distribution of the geo-locations
zones_visited = np.unique(agent_location.values)
# row_idx, col_idx, cell_lat, cell_lon = mygeo.zone_id_to_rowcol_latlon(zones_visited,
#                                n_rows, n_cols, lat_min, lon_min, CELL)

# plt.scatter(cell_lon, cell_lat, s = 0.05)

#### Identify staying and passing zones


# def rle_lengths_by_value_1d(arr):
#     """
#     Run-length encode a 1D integer array and aggregate lengths by value.

#     Parameters
#     ----------
#     arr : 1D array-like of int
#         Sequence of zone indices for one agent over time.

#     Returns
#     -------
#     out : dict[int, np.ndarray]
#         Mapping: zone_id -> np.array of consecutive lengths (in steps).
#         Example: [1,2,2,2,2,2,3,3,3,2] -> {1:[1], 2:[5,1], 3:[3]}
#     """
#     a = np.asarray(arr)
#     if a.size == 0:
#         return {}

#     # Find starts of runs
#     # mask[i] == True when i is the start of a run
#     # (first position or value != previous)
#     mask = np.empty(a.size, dtype=bool)
#     mask[0] = True
#     if a.size > 1:
#         mask[1:] = a[1:] != a[:-1]

#     run_starts = np.flatnonzero(mask)
#     run_values = a[run_starts]
#     run_lengths = np.diff(np.append(run_starts, a.size))

#     # Aggregate lengths by value
#     out = {}
#     for v, L in zip(run_values, run_lengths):
#         v = int(v)
#         if v in out:
#             out[v].append(int(L))
#         else:
#             out[v] = [int(L)]
#     # Convert lists to compact arrays
#     for k in out:
#         out[k] = np.asarray(out[k], dtype=np.int32)
#     return out

# def compute_stay_lengths_by_agent(agent_location: pd.DataFrame) -> dict:
#     """
#     For each agent (row), compute consecutive stay lengths per zone.

#     Parameters
#     ----------
#     agent_location : DataFrame (n_agents x n_steps)
#         Each row corresponds to one agent; each column is a 5-minute time step.
#         Cell values are integer zone indices.

#     Returns
#     -------
#     result : dict
#         Mapping: agent_key -> {zone_id: np.array of lengths}
#         - agent_key is whatever is in agent_location.index (can be int/tuple/MultiIndex key).
#     """
#     values = agent_location.to_numpy()
#     agents = agent_location.index
#     result = {}

#     for i, agent_key in enumerate(agents):
#         result[agent_key] = rle_lengths_by_value_1d(values[i])

#     return result


agent_zone_staying_times = mygeo.compute_stay_lengths_by_agent(agent_location)
staying_zones_per_agent = {key: pd.Series({key2: np.mean(val2) \
                                           for key2, val2 in val.items() if np.mean(val2) > 1}) \
 for key, val in agent_zone_staying_times.items()}

passing_zones_per_agent = {key: pd.Series({key2: np.mean(val2) \
                                           for key2, val2 in val.items() if np.mean(val2) == 1}).index.tolist() \
 for key, val in agent_zone_staying_times.items()}       
del agent_zone_staying_times
  
#### Print out the distribution of number of staying zones by duration bucket
levels = {
    ">=8h": lambda v: v >= 96,                      # >= 8h
    "4h-8h": lambda v: (v >= 48) & (v < 96),        # 4h–8h
    "2h-4h": lambda v: (v >= 24) & (v < 48),        # 2h–4h
    "0-2h":  lambda v: (v >= 0) & (v < 24),         # <2h
}

results = {}
for label, cond in levels.items():
    counts = pd.Series({key: len(val[cond(val)]) 
                        for key, val in staying_zones_per_agent.items()})
    results[label] = counts.value_counts().sort_index()
df_results = pd.DataFrame(results).fillna(0).astype(int)
print(df_results)

####
# staying_zones_per_agent: dict[agent] -> dict[zone_id] -> np.array(lengths_in_steps)
bucket_zones = {label: set() for label in levels}

len(list(set(v for values in passing_zones_per_agent.values() for v in values)))

for agent_dict in staying_zones_per_agent.values():
    for zone_id, lens in agent_dict.items():
        a = np.atleast_1d(lens)          # ensure array
        if a.size == 0:
            continue
        a = a[~np.isnan(a)]              # drop NaNs if any
        if a.size == 0:
            continue
        # lengths should already be ints (counts of 5-min steps)
        a = a.astype(int, copy=False)
        for label, cond in levels.items():
            if cond(a).any():
                bucket_zones[label].add(int(zone_id))

# Optional: convert sets to sorted lists
bucket_zones = {k: sorted(v) for k, v in bucket_zones.items()}

# Quick summary counts
summary = pd.Series({k: len(v) for k, v in bucket_zones.items()}).sort_index()
print(summary)

''' Contigency table of different types of staying zones'''
# Convert to sets for easy intersection
zone_sets = {k: set(v) for k, v in bucket_zones.items()}

# Concatenate all staying zones into a single array
staying_zones = np.concatenate([np.asarray(sub) 
                                for sub in bucket_zones.values()])

staying_zones = np.unique(staying_zones)

# All unique bucket names
buckets = list(zone_sets.keys())

# Build an empty DataFrame
contingency = pd.DataFrame(0, index=buckets, columns=buckets)

# Fill the table with intersections
for i in buckets:
    for j in buckets:
        contingency.loc[i, j] = len(zone_sets[i] & zone_sets[j])

print("Contingency Table:")
print(contingency)

''' Spatial plot of staying zones in different categories '''
levels_order = list(levels.keys())
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

# Optional: fixed axis limits so all subplots share the same view
lat_min_plot = global_bounds['lat_min']
lat_max_plot = global_bounds['lat_max']
lon_min_plot = global_bounds['lon_min']
lon_max_plot = global_bounds['lon_max']

for ax, label in zip(axes, levels_order):
    zones = bucket_zones.get(label, [])
    if len(zones) > 0:
        zone_loc = mygeo.zone_id_to_rowcol_latlon(
            np.asarray(zones), n_rows, n_cols, lat_min, lon_min, CELL
        )
        # Your current convention: x=latitude (zone_loc[2]), y=longitude (zone_loc[3])
        ax.scatter(zone_loc[2], zone_loc[3], s=0.3)

        # If you prefer geographic convention (x=lon, y=lat), use:
        # ax.scatter(zone_loc[3], zone_loc[2], s=0.3)
    else:
        ax.text(0.5, 0.5, "No zones", ha="center", va="center", fontsize=12)
    
    ax.set_title(f"Spatial Distribution of {label} Zones", fontsize=11)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_xlim(lat_min_plot, lat_max_plot)   # keep same extent across subplots
    ax.set_ylim(lon_min_plot, lon_max_plot)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.3, alpha=0.4)

plt.tight_layout()
plt.show()

'''Plot the staying and passing zones together'''
# staying zones (already combined across buckets)
staying_zones = np.concatenate([np.asarray(sub) 
                                for sub in bucket_zones.values()]).tolist()
staying_zones = np.unique(staying_zones)   # optional: drop duplicates
staying_zones_loc = mygeo.zone_id_to_rowcol_latlon(
    staying_zones, n_rows, n_cols, lat_min, lon_min, CELL
)

# passing zones (Note that this is not real passing zone, 
# since a zone can be both passing and staying)
passing_zones = zones_visited[~np.isin(zones_visited, staying_zones)]
passing_zones_loc = mygeo.zone_id_to_rowcol_latlon(
    passing_zones, n_rows, n_cols, lat_min, lon_min, CELL
)

# plot both
plt.figure(figsize=(10, 8))
plt.scatter(passing_zones_loc[3], passing_zones_loc[2], 
            s=0.05, color='red', label='Passing Zones')
plt.scatter(staying_zones_loc[3], staying_zones_loc[2], 
            s=0.3, color='blue', label='Staying Zones')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Staying vs Passing Zones")
plt.legend(markerscale=20, fontsize=9)   # markerscale makes tiny points visible in legend
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linewidth=0.3, alpha=0.4)
plt.show()



''' Try Clustering the >=8h staying zones'''

out = mygeo.zone_id_to_rowcol_latlon(
    np.array(bucket_zones['>=8h']), n_rows, n_cols, lat_min, lon_min, CELL
)

zone_loc_stay = pd.DataFrame(out[2:]).T
zone_loc_stay.columns = ['lat', 'lon']

plt.scatter(zone_loc_stay['lon'], zone_loc_stay['lat'], s = 0.1)

coords_deg = np.c_[zone_loc_stay['lat'], zone_loc_stay['lon']]
coords_rad = np.radians(coords_deg)

# eps in radians (1 km on Earth's radius)
eps_km = 1.0
eps_rad = eps_km / 6371.0088  # Earth radius km

db = DBSCAN(eps=eps_rad, min_samples=5,\
            metric='haversine').fit(coords_rad)
labels = db.labels_

zone_loc_stay['cluster'] = labels

# 画散点图，每个 cluster 用不同颜色
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    zone_loc_stay['lon'], 
    zone_loc_stay['lat'], 
    c=zone_loc_stay['cluster'], 
    cmap='tab20',   # 20种颜色
    s=0.5,            # 点大小
    alpha=0.7
)

# 给每个 cluster 加上 label 文字（在 cluster 中心）
for cluster_id in np.unique(labels):
    cluster_points = zone_loc_stay[zone_loc_stay['cluster'] == cluster_id]
    center_lon = cluster_points['lon'].mean()
    center_lat = cluster_points['lat'].mean()
    plt.text(center_lon, center_lat, str(cluster_id), fontsize=8,
             ha='center', va='center', weight='bold')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("DBSCAN Clusters (1 km eps)")
plt.colorbar(scatter, label="Cluster ID")
plt.show()
















# agent_location_df.to_parquet('all_agents/str_data/agent_location_str_100m.parquet')
# agent_location.index = pd.Index([tuple(x) for x in agent_location.index])

zone_centers = pd.read_parquet('str_data/zone_centers.parquet').set_index('index')

unique_zones = np.unique(agent_location.values)

# 给每个点标记 day_idx
def add_day_idx(df_time_col: pd.Series) -> pd.Series:
    t0 = df_time_col.dt.normalize().iloc[0]
    return (df_time_col.dt.normalize() - t0).dt.days

# 给每个点分配时间段标签
def add_time_block(df_time_col: pd.Series) -> pd.Series:
    hour = df_time_col.dt.hour
    bins = [0, 6, 12, 18, 24]
    labels = [0, 1, 2, 3]  # 四个时段
    return pd.cut(hour, bins=bins, labels=labels, right=False).astype(int)

# 配色方案：从亮到暗
time_colors = {
    0: "#FFD700",  # 0–6h, bright yellow
    1: "#FF8C00",  # 6–12h, orange
    2: "#DC143C",  # 12–18h, crimson
    3: "#4B0082"   # 18–24h, indigo
}

out_dir = "trajectory_EDA"
os.makedirs(out_dir, exist_ok=True)

# ===== 主循环 =====
for agent in range(50):
    # 1) 构造轨迹数据（经度=longitude 作为 x，纬度=latitude 作为 y）
    temp_trajectory = zone_centers.loc[agent_location.loc[(agent, 0), :].values].copy()
    temp_trajectory['time'] = pd.to_datetime(agent_location.columns)
    temp_trajectory['day_idx'] = add_day_idx(temp_trajectory['time'])
    temp_trajectory['time_block'] = add_time_block(temp_trajectory['time'])

    # 2) 统一坐标范围（整份 PDF 内一致）
    lon = temp_trajectory['longitude'].to_numpy()
    lat = temp_trajectory['latitude'].to_numpy()
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_pad = (lon_max - lon_min) * 0.05 if lon_max > lon_min else 0.001
    lat_pad = (lat_max - lat_min) * 0.05 if lat_max > lat_min else 0.001
    x_min, x_max = lon_min - lon_pad, lon_max + lon_pad
    y_min, y_max = lat_min - lat_pad, lat_max + lat_pad

    # 3) 生成 PDF（7x4 子图布局；可能两页）
    out_path = os.path.join(out_dir, f"trajectories_pop{agent}_agent0.pdf")
    with PdfPages(out_path) as pdf:
        nrows, ncols = 4, 7
        plots_per_page = nrows * ncols

        for page_start in range(0, 29, plots_per_page):
            fig, axes = plt.subplots(nrows, ncols, figsize=(28, 10))
            axes = axes.flatten()

            days_this_page = list(range(page_start, min(page_start + plots_per_page, 29)))
            for ax_i, day in enumerate(days_this_page):
                ax = axes[ax_i]
                subset = temp_trajectory.loc[temp_trajectory['day_idx'] == day].sort_values('time')
                if subset.empty:
                    ax.set_title(f"day {day} (no data)", fontsize=8)
                    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                    ax.set_aspect('equal', adjustable='box')
                    ax.tick_params(labelsize=6)
                    continue

                # —— 分时段绘制：连线 + 小点 ——
                for block, group in subset.groupby('time_block'):
                    g = group.sort_values('time').copy()
                    if g.empty:
                        continue

                    # 连线
                    ax.plot(g['longitude'], g['latitude'],
                            color=time_colors[block], linewidth=1)

                    # 小点（所有点）
                    ax.scatter(g['longitude'], g['latitude'],
                               color=time_colors[block], s=4, alpha=0.7)

                    # —— 连续“停留点”高亮（同一经纬度连续 >=5 行）——
                    eps = 1e-6  # 浮点容差，避免误判
                    same_as_prev = (
                        np.isclose(g['longitude'], g['longitude'].shift(), atol=eps) &
                        np.isclose(g['latitude'],  g['latitude'].shift(),  atol=eps)
                    )
                    change = ~same_as_prev
                    grp = change.cumsum()
                    run_len = g.groupby(grp)['longitude'].transform('size')
                    is_stay = run_len >= 5
                    if is_stay.any():
                        gs = g.loc[is_stay]
                        ax.scatter(gs['longitude'], gs['latitude'],
                                   s=18, facecolor=time_colors[block],
                                   edgecolor='k', linewidth=0.6, alpha=0.95, zorder=3)

                # 坐标轴和比例
                ax.set_title(f"day {day}", fontsize=8)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect('equal', adjustable='box')
                ax.tick_params(labelsize=6)

            # 删除多余子图
            for j in range(len(days_this_page), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

# agent_location = np.loadtxt('str_data/agent_location_str.txt')
# agent_location = agent_location.astype('int')
# unique_zones = np.unique(agent_location)


# unique_agents = np.arange(agent_location.shape[0])


# zoneid_map = {unique_zones[i]:i for i in range(len(unique_zones))}
# agentid_map =  {unique_agents[i]:i for i in range(len(unique_agents))}


Nzones = len(unique_zones)
Nagents = agent_location.shape[0]
Ntimes = agent_location.shape[1]

pd.read_csv('id_map.csv')

# zone_centers = all_agent_df.groupby('index')[['latitude', 'longitude']].mean()
# zone_centers.rename(index=zoneid_map, inplace=True)
# zone_centers = np.array(zone_centers.iloc[np.argsort(zone_centers.index),:])
# np.savetxt('str_data/zone_centers_str.txt', zone_centers)
zone_centers = np.loadtxt('str_data/zone_centers_str.txt')
# st = time.time()
# agent_location = np.vectorize(zoneid_map.get)(agent_location)
# ed = time.time()
# ed - st

# np.savetxt('str_data/agent_location_str.txt', agent_location, fmt='%d')
# agent_location = agent_location.astype('int')


occupancy_counts = scipy.sparse.lil_matrix((Nzones, Ntimes), dtype=int)

# Process each time point
for t in range(Ntimes):
    if t % 100 == 0:
        print(t)
    counts_t = np.unique(agent_location[:, t], return_counts=True)
    for zone, count in zip(counts_t[0], counts_t[1]):
        occupancy_counts[zone, t] = count
occupancy_counts = occupancy_counts.tocsr()
sps.save_npz('str_data/occupancy_counts_str.npz', occupancy_counts, compressed=True)

occupancy_counts = sps.load_npz('str_data/occupancy_counts_str.npz')



def generateMatricesSparse(listOfSequences, numCells):
    # listOfSequences = agent_location
    # numCells = Nzones
    numTimestamps = len(listOfSequences[0])
    num_Agents = len(listOfSequences)
    matrixList = [None] * (numTimestamps - 1)
    for t in range(numTimestamps - 1):
        if t % 10 == 0:
            print(t)
        mat = lil_array((numCells, numCells))
        for a in range(num_Agents):
            i = listOfSequences[a][t]
            j = listOfSequences[a][t + 1]
            mat[i, j] = mat[i, j] + 1
        matrixList[t] = mat.tocsr()
    return matrixList

def generateMatricesSparse2(listOfSequences, numCells):
    # listOfSequences = agent_location
    # numCells = Nzones
    numTimestamps = len(listOfSequences[0])
    num_Agents = len(listOfSequences)
    matrixList = [None] * (numTimestamps - 1)
    for t in range(numTimestamps - 1):
        if t % 1000 == 0:
            print(t)
        print(t)
        # Initialize lists to store data for coo_matrix
        data = []
        row = []
        col = []

        direction, counts = np.unique(listOfSequences[:, t:t+2], axis=0, return_counts=True)
        row_idx, col_idx = direction.T
        
        # Append data for coo_matrix
        data.extend(counts)
        row.extend(row_idx)
        col.extend(col_idx)
        
        # Create a coo_matrix
        mat_coo = scipy.sparse.coo_matrix((data, (row, col)), shape=(numCells, numCells), dtype=int)
        
        # Convert to CSR format and store in the list
        matrixList[t] = mat_coo.tocsr()
        
        
    return matrixList

def sum_sparse_matrices(matrixList):
    if not matrixList:
        return None

    # Initialize an empty sparse matrix with the same shape as the matrices in the list
    total_matrix = scipy.sparse.csr_matrix(matrixList[0].shape)

    for matrix in matrixList:
        if matrix is not None:
            total_matrix += matrix

    return total_matrix

def save_sparse_matrix_list(matrixList, filename):
    matrix_dict = {f'matrix_{i}': matrix for i, matrix in enumerate(matrixList)}
    
    # Save all sparse matrices to a single .npz file
    scipy.sparse.save_npz(filename, scipy.sparse.hstack([scipy.sparse.coo_matrix(matrix) \
                                                         for matrix in matrixList]), compressed=True)

def load_sparse_matrix_list(filename):
    # Load the .npz file
    sparse_data = sp.load_npz(filename)
    
    # Assuming we know the shape of each matrix, we can split them accordingly
    # Here, we need the shapes to correctly split the hstacked data
    matrixList = []
    start_col = 0
    for shape in shapes:  # shapes should be a list of tuples with the shape of each matrix
        end_col = start_col + shape[1]
        matrix = sparse_data[:, start_col:end_col].tocsr()
        matrixList.append(matrix)
        start_col = end_col

    return matrixList




batch_num = 8; batch_size = int(agent_location.shape[1] / batch_num)
total_matrix = scipy.sparse.csr_matrix((Nzones, Nzones))
for i in range(batch_num):
    # i = 7
    trans_mat_i = generateMatricesSparse2(agent_location[:, batch_size*i: (batch_size*(i+1))], Nzones)

    save_sparse_matrix_list(trans_mat_i, f'str_data/trans_mat/transmat_{i}.npz' )
    
    total_matrix += sum_sparse_matrices(trans_mat_i)
    
save_sparse_matrix_list(total_matrix, f'str_data/trans_mat/sum_transmat.npz' )



# sanity check if temporal transition matrix matches the occupancy (passed)
# for t in range(len(trans_mat_all)):
#     if t % 10 == 0:
#         print(t)
#     trans_mat_t = trans_mat_all[t]
#     trans_mat_t = trans_mat_t.toarray()
#     x = np.sum(trans_mat_t, axis = 1)
#     if not all(occupancy_counts[:,t] == x):
#         print("bad")


########### Transition-flow based Hierarchical Spatial Clustering #############
sum_flow_mat = lil_array((Nzones, Nzones)).tocsr()
for i in range(len(trans_mat_all)):
    sum_flow_mat += trans_mat_all[i]

sum_flow_mat = sum_flow_mat.toarray()

upper_tri_values = np.max(np.hstack([sum_flow_mat[np.triu_indices(Nzones, k = 1)].reshape(-1,1), \
                                     sum_flow_mat.T[np.triu_indices(Nzones, k = 1)].reshape(-1,1)]), axis = 1)
sum_flow_mat[np.triu_indices(Nzones, k = 1)] = upper_tri_values
sum_flow_mat.T[np.triu_indices(Nzones, k = 1)] = upper_tri_values



zone_dist_mat = distance_matrix(zone_centers, zone_centers) * 111
sum_flow_mat[zone_dist_mat > np.median(zone_dist_mat)] = 2*np.max(sum_flow_mat)
np.fill_diagonal(sum_flow_mat, np.zeros((Nzones, )))
condensed_dist_matrix = squareform(sum_flow_mat, force='tovector')
linked = linkage(condensed_dist_matrix, method='ward')  # You can choose other methods like 'single', 'complete', etc.


desired_num_clusters = 100
high_levels_zones = fcluster(linked, desired_num_clusters, criterion='maxclust') - 1
high_zone_roster = [np.where(high_levels_zones == i)[0] for i in range(desired_num_clusters)]

agent_location_high = high_levels_zones[agent_location]

# np.savetxt('agent_location_high_' + str(desired_num_clusters) + '_sf.txt', agent_location_high, fmt='%d')
agent_location_high = np.loadtxt('str_data/agent_location_high_' + str(desired_num_clusters) + '_str.txt')
agent_location_high = agent_location_high.astype('int')

# for i in list(range(1, 6)) + [10, 20]:
#     # i  = 10
#     desired_num_clusters_i = i
    
#     tmp_cluster = fcluster(linked, desired_num_clusters_i, criterion='maxclust') - 1
#     plt.scatter(zone_centers[:, 0], zone_centers[:, 1], c=tmp_cluster, s =0.5,cmap='tab20')
#     ax = plt.gca()  # Get current axis
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6)) 
#     plt.title('Sptaial Clustering with ' + str(desired_num_clusters_i) + ' of Clusters')
#     plt.savefig(f"trial2/figures/spatial_clustering_{i}.pdf")
#     plt.close()



# # # Proportion of total transitions crossing higher-level zones
# high_vs_fine = np.sum(high_levels_zones[agent_location][:, 1:] != high_levels_zones[agent_location][:, :-1]) / \
#     np.sum(agent_location[:, 1:] != agent_location[:, :-1])
    
# a = np.zeros((190, ))
# i = 0
# for nclusters in range(10, 200):
#     temp = fcluster(linked, nclusters, criterion='maxclust') - 1
#     a[i] = np.sum(temp[agent_location][:, 1:] != temp[agent_location][:, :-1]) / \
#         np.sum(agent_location[:, 1:] != agent_location[:, :-1])
#     i+=1
# plt.plot(list(range(10, 200)), a)
# plt.title('Proportion of Transition Captured by High-level Zones')
# plt.xlabel('number of high-level zones')
# plt.ylabel('prop%')
# plt.savefig(f"trial2/figures/trans_prop.pdf")
# plt.show()

# np.arange(10, 200)[np.where(a > 0.9)[0]]
# # proportion of non-staying transitions at finer-level zones over all transitions based on 
# # rectangular-clustering
# np.mean(high_levels_zones[agent_location][:, 1:] != high_levels_zones[agent_location][:, :-1])

################### Transitions at higher-level zones ########################
occupancy_counts_high = np.zeros((desired_num_clusters, Ntimes))
for t in range(Ntimes):
    temp_t = np.unique(agent_location_high[:,t], return_counts = True)

    occupancy_counts_high[:,t][temp_t[0]] = temp_t[1]
    
unique_flows = np.unique(np.hstack([agent_location_high[:,:-1].reshape(-1, 1), \
                                    agent_location_high[:,1:].reshape(-1, 1)]), axis = 0)
    

# file_name = 'transition_flows_' +  str(desired_num_clusters) + '_sf.pkl'
# if os.path.isfile(file_name):
#     with open(file_name, 'rb') as file:
#         # Deserialize and load the list using pickle.load
#         transition_flows = pickle.load(file)
# else:
#     # Get the transition flow data for each direction over time
#     transition_flows = np.zeros((unique_flows.shape[0], Ntimes-1))
#     for t in range(Ntimes-1):
#         print(t)
#         # trans_mat_t = trans_mat_all[t].toarray()
#         for i in range(len(unique_flows)):
#             # i = 1805
#             selected_rows = trans_mat_all[t][high_zone_roster[unique_flows[i, 0]]]
#             n_columns = trans_mat_all[t].shape[1]
#             column_selector = eye(n_columns, format='csr')
#             column_selector = column_selector[:, high_zone_roster[unique_flows[i, 1]]]
#             transition_flows[i, t]  = selected_rows.dot(column_selector).sum()
#             # transition_flows[i, t] = np.sum(trans_mat_t[high_zone_roster[unique_flows[i, 0]], :][:, high_zone_roster[unique_flows[i, 1]]])

# transition_flows = sps.csr_matrix(transition_flows)
# sps.save_npz('trial2/trans_flow_high_sf.npz', transition_flows, compressed=True)
transition_flows = sps.load_npz('str_data/trans_flow_high_str.npz')
transition_flows = transition_flows.toarray()


# Reorder the destinations based on empirial transition probability averaged over time
destinations_by_zone = [np.where(unique_flows[:,0] == i)[0] for i in range(desired_num_clusters)]
for i in range(desired_num_clusters):
    # i = 76
    temp = occupancy_counts_high[i,:-1]
    temp[temp == 0] = 0.1
    trans_probs_i = transition_flows[destinations_by_zone[i], :] / temp
    destinations_by_zone[i] = destinations_by_zone[i][np.argsort(-np.mean(trans_probs_i, axis = 1))]
    

unique_flows = unique_flows[np.concatenate(destinations_by_zone), :]


# with open('str_data/unique_flows_' + str(desired_num_clusters) + '_str.pkl', 'wb') as file:
#     pickle.dump(unique_flows, file)
    

flow_list = [transition_flows[destinations_by_zone[i], :] for i in range(desired_num_clusters)]

# Here we get all series independently to implement on DCC
input_list_trans = []
for i in range(desired_num_clusters):
    # i = 0
    flow_list_i = flow_list[i]
    n_i = occupancy_counts_high[i,:-1] - np.vstack([np.zeros((1, flow_list_i.shape[1]))] + \
                                      [np.cumsum(flow_list_i[:-1, :], axis = 0)])
    input_list_trans += [np.concatenate([n_i[j, :], flow_list[i][j, :]]) for j in range(len(flow_list_i))]

input_trans_flow = np.array(input_list_trans)
input_trans_flow = sps.csr_matrix(input_trans_flow)

# sps.save_npz('input_trans_flow_' + str(desired_num_clusters) + '_sf.npz', input_trans_flow, compressed=True)



####################### Location of each higher-level zone ########################

occupancy_counts_fine = sps.load_npz('trial2/occupancy_counts_sf.npz')
occupancy_counts_fine = occupancy_counts_fine.toarray()
occupancy_counts_high[occupancy_counts_high == 0.1] = 0

sorted_high_zone_roster = copy.deepcopy(high_zone_roster)
for i in range(desired_num_clusters):
    # i = 76
    
    occupancy_i = occupancy_counts_fine[high_zone_roster[i], :]
    prop_i = np.zeros((len(occupancy_i), ))
    temp_idx = np.where(occupancy_counts_high[i, :] != 0)[0]
    prop_i = np.mean(occupancy_i[:, temp_idx] / occupancy_counts_high[i, temp_idx], axis=1)
    ordered_idx = np.argsort(-prop_i)
    
    # We pick the first x zones whose averaged cumulative proportion is greater than 0.98
    sorted_high_zone_roster[i] = high_zone_roster[i][ordered_idx[:np.where(np.cumsum(prop_i[ordered_idx]) > 0.98)[0][0]+1]]

subzone_loc_list = [occupancy_counts_fine[sorted_high_zone_roster[i], :] for i in range(desired_num_clusters)]
    

# Here we get all series independently to implement on DCC
input_list_trans2 = []
for i in range(desired_num_clusters):
    # i = 0
    subzone_loc_list_i = subzone_loc_list[i]
    
    n_i = occupancy_counts_high[i,:] - np.vstack([np.zeros((1, subzone_loc_list_i.shape[1]))] + \
                                      [np.cumsum(subzone_loc_list_i[:-1, :], axis = 0)])
    
    input_list_trans2 += [np.concatenate([n_i[j, :], subzone_loc_list[i][j, :]]) for j in range(len(subzone_loc_list_i))]
    
input_loc_dist = np.array(input_list_trans2)
input_loc_dist = sps.csr_matrix(input_loc_dist)

# sps.save_npz('input_loc_dist_' + str(desired_num_clusters) + '_sf.npz', input_loc_dist, compressed=True)

def count_files(directory):
    entries = os.listdir(directory)
    count = sum(os.path.isfile(os.path.join(directory, entry)) for entry in entries)
    return count


# unique_flows = np.loadtxt('unique_flows_100_str.txt').astype(int)

with open('str_data/unique_flows_' + str(desired_num_clusters) + '_str.pkl', 'rb') as file:
    unique_flows = pickle.load(file)

results_transition = list()
for i in range(count_files('results_transition_str')):
    with open('results_transition_str/results_transition' + str(i) +  '.pkl', 'rb') as file:
        # Serialize and save the list using pickle.dump
        a = pickle.load(file)
        results_transition += a

# [i for i, x in enumerate(results_transition) if np.sum(np.isnan(x)) > 0]

results_transition[8] = np.vstack(out[:2]).reshape(-1,)

lag = 5
fitted_cascade_probs = np.vstack(results_transition)
fitted_probs = []
for i in range(desired_num_clusters):
    # i = 0
    idx_i = np.where(unique_flows[:,0] == i)[0]
    temp_probs = fitted_cascade_probs[idx_i[:-1], :]
    
    fitted_probs_i = np.vstack([np.ones((1, temp_probs.shape[1])), \
                                np.cumprod(1 - temp_probs, axis = 0)]) * \
        np.vstack([temp_probs, np.ones((1, temp_probs.shape[1]))])
    fitted_probs.append(fitted_probs_i)

fitted_probs = np.vstack(fitted_probs)



fitted_probs2 = copy.deepcopy(fitted_probs)
fitted_probs2[(0 < fitted_probs2) & (fitted_probs2 < 1e-5)] = 0

full_trans_probs = [np.zeros((desired_num_clusters, desired_num_clusters, fitted_probs.shape[1]-k + 1)) for k in range(1, lag)]

for i in range(len(unique_flows)):
    # i = -1
    full_trans_probs[0][unique_flows[i, 0], unique_flows[i, 1], :] = fitted_probs2[i, :]

for k in range(2, lag):
    for i in range(desired_num_clusters):
        for j in range(desired_num_clusters):
            full_trans_probs[k-1][i, j, :] = np.sum(full_trans_probs[k-2][i, :, :-1] * full_trans_probs[0][:, j, (k-1):], axis = 0)


# full_trans_probs0 = full_trans_probs[0].reshape(desired_num_clusters, desired_num_clusters*full_trans_probs[0].shape[2])
# full_trans_probs0_back = full_trans_probs0.reshape(desired_num_clusters, desired_num_clusters, int(full_trans_probs0.shape[1] / full_trans_probs0.shape[0]))
# sparse_trans_prob = csr_matrix(full_trans_probs0)
# scipy.sparse.save_npz('trial2/matrix_tmp.npz', sparse_trans_prob)

with open('full_trans_probs_' + str(desired_num_clusters) + '_str.pkl', 'wb') as file:
    pickle.dump(full_trans_probs, file) 

# with open('full_trans_probs_' + str(desired_num_clusters) + '_str.pkl', 'rb') as file:
#     full_trans_probs = pickle.load(file) 


# results_location = list()
# for i in range(count_files('results_location_sf_04-18-24')):
#     with open('results_location_sf_04-18-24/results_location' + str(i) +  '.pkl', 'rb') as file:
#         # Serialize and save the list using pickle.dump
#         a = pickle.load(file)
#         results_location += a
        
    
# fitted_cascade_prob_loc = [None] * desired_num_clusters
# for i in range(desired_num_clusters):
#     # i = 0
#     fitted_cascade_prob_loc[i] = np.vstack(results_location[:len(sorted_high_zone_roster[i])])
#     results_location = results_location[len(sorted_high_zone_roster[i]):]
    

# fitted_probs_loc = []
# for i in range(desired_num_clusters):
#     # i = 0
    
    
#     temp_probs = fitted_cascade_prob_loc[i]
    
#     fitted_probs_tmp = np.vstack([np.ones((1, temp_probs.shape[1])), \
#                                 np.cumprod(1 - temp_probs, axis = 0)]) * \
#         np.vstack([temp_probs, np.ones((1, temp_probs.shape[1]))])

#     fitted_probs_i = dict()
#     for j in range(len(sorted_high_zone_roster[i])):
#         fitted_probs_i[sorted_high_zone_roster[i][j]] = fitted_probs_tmp[j,:]
#     fitted_probs_loc.append([high_zone_roster[i], fitted_probs_i])



# with open('results_loc_' + str(desired_num_clusters) + '_sf.pkl', 'wb') as file:
#     pickle.dump(fitted_probs_loc, file)   
    
    
    
    
    