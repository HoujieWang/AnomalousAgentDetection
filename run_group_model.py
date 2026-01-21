#!/usr/bin/env python
# coding: utf-8

# In[1]:
''' 
    This file reads saved agent location in discrete zone index, and run group-level model and generate forecasts as feature for individual-level model.

'''

import Utilities2025.geo_utilities as mygeo
import Utilities2025.clustering_utilities as mycl
import Utilities2025.counting_utilities as mycount
import Utilities2025.run_model_utilities as myrun
import Utilities2025.staying_feature_utilities as mystay_x

import pandas as pd 
import numpy as np
import os
import pyarrow
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import xticks
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from IPython.display import clear_output

from collections import deque
from sklearn.neighbors import BallTree
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.special import logit
import scipy
from typing import Optional, Union, List, Dict, Tuple
from pybats.dglm import bin_dglm
import time

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score


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

threshold = 0.99


agent_location_cl = pd.read_parquet(f"agent_location_cl_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.parquet", engine="pyarrow")
agent_location_cl.columns = pd.to_datetime(agent_location_cl.columns)
# %%


M, row_idx, time_cols = mycount.timewise_transition_counts_sparse_streaming(
    agent_location_cl.iloc[:,:], lag=1, include_self=True
)
M, row_idx = mycount.reorder_rows_within_origin_by_totals(M, row_idx,
                                                origin_asc=True,
                                                totals_desc=True)

transition_counts = pd.DataFrame.sparse.from_spmatrix(M, index=row_idx, columns=time_cols)
occupancy_counts = mycount.zone_occupancy_counts(agent_location_cl)
occupancy_counts.columns = pd.to_datetime(occupancy_counts.columns)

size_counts = mycount.binomial_cascade_sizes_fast(transition_counts, occupancy_counts)
size_counts.columns = pd.to_datetime(size_counts.columns)


transition_counts, size_counts = mycount.filter_by_prefix_threshold(
    M, row_idx,
    transition_counts,
    size_counts,
    occupancy_counts,
    threshold=threshold
)

# %%
# origin_idx = -170
time_ind = myrun.make_simple_time_indicators(transition_counts.columns, day_start="11:00", day_end="00:00")

st = time.time()
kernel_prob_df = myrun.kernel_smoothed_transition_probs(
    # transition_counts.loc[[origin_idx]], size_counts.loc[[origin_idx]],
    transition_counts, size_counts,
    freq="5min",
    h_minutes=5,
    window_minutes=15,
    include_same_day= True
)
end = time.time()
end - st
kernel_prob_df.fillna(0.5, inplace=True)
kernel_prob_df_logit = logit(kernel_prob_df.copy())
kernel_prob_df_logit.to_parquet(f"temp_feature_storage/kernel_prob_df_logit_{threshold}_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.parquet")


# usage:
kernel_multiprob_df = myrun.stick_to_mult_by_group(kernel_prob_df, myrun.stick_to_mult_matrix)
st = time.time()
kernel_occu_df = myrun.kernel_smoothed_occupancy_counts(
    # occupancy_counts.loc[[origin_idx]],
    occupancy_counts,
    freq="5min",
    h_minutes=5,
    window_minutes=15,
    include_same_day= True
)
end = time.time()
end - st
occupancy_ratio = pd.DataFrame(occupancy_counts.values[:,1:] / kernel_occu_df.values[:,:-1],
                               index = occupancy_counts.index, columns = kernel_multiprob_df.columns)

i_index = kernel_multiprob_df.index.get_level_values(0)
occ_expanded = occupancy_ratio.reindex(i_index)  # duplicates each i for its j's
occ_expanded = occ_expanded.reindex(columns=kernel_multiprob_df.columns)

scaled = kernel_multiprob_df * occ_expanded.to_numpy()
kernel_occu_change_df = np.log1p(scaled)
kernel_occu_change_df.to_parquet(f"temp_feature_storage/kernel_occu_change_df_{threshold}_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.parquet")

# %%

st = time.time()
models, fcasts, states = myrun.fit_binomial_dglms(
    trans_counts = transition_counts,
    n_counts = size_counts,
    daily_harmonics=1,
    weekly_harmonics=1,
    deltrend=1,
    delseason=1,
    delregn=1,
    pair_features_list = [kernel_prob_df_logit, kernel_occu_change_df],
    time_features_list = [time_ind[['ind_is_day', 'ind_is_weekday']].T]
)
end = time.time()
end - st

with open(f"models_{threshold}_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.pkl", "wb") as f:
    pickle.dump(models, f)

with open(f"fcasts_{threshold}_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.pkl", "wb") as f:
    pickle.dump(fcasts, f)
    
with open(f"states_{threshold}_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.pkl", "wb") as f:
    pickle.dump(states, f)
    