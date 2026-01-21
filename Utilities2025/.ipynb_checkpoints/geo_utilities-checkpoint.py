'''
This file contains some geo-location related utilitiy functions
'''
import numpy as np
import pandas as pd
global_bounds = {'lat_min': 35.51009030444566, \
                 'lat_max': 36.5818509, \
                 'lon_min': -84.95347326088375, \
                 'lon_max': -83.70374631780497}

def latlon_to_index(lat, lon, lat_min, lon_min, n_rows, n_cols, CELL):
    """
    Map latitude/longitude coordinates to a 1D zone index on a fixed grid.

    The grid is defined by:
      - Origin at (lat_min, lon_min)
      - Cell size = CELL (in degrees, e.g. 0.001 ≈ 111 m latitude resolution)
      - n_rows = number of rows (latitude direction, south → north)
      - n_cols = number of columns (longitude direction, west → east)

    The flattened zone index is computed as:
        zone_id = row * n_cols + col
    where:
        row = floor((lat - lat_min) / CELL)
        col = floor((lon - lon_min) / CELL)

    Parameters
    ----------
    lat : float or array-like
        Latitude(s) in degrees.
    lon : float or array-like
        Longitude(s) in degrees.

    Returns
    -------
    zone_id : int or ndarray of int
        Flattened 1D zone index (0 ≤ zone_id < n_rows * n_cols).
        - If inputs are scalars, returns an int.
        - If inputs are arrays, returns a NumPy ndarray of ints.

    Notes
    -----
    - Values are clipped to valid ranges [0, n_rows-1] and [0, n_cols-1].
    - Resolution depends on CELL: 
        * CELL = 0.001 → ~111 m latitude cell size.
        * Longitude resolution decreases with latitude (cosine factor).

    Examples
    --------
    >>> latlon_to_index(36.024971, -84.211232)
    452731   # Example flattened index for that coordinate

    >>> latlon_to_index([36.0, 36.1], [-84.2, -84.3])
    array([452100, 453450])
    """
    r = np.floor((lat - lat_min) / CELL).astype(int)
    c = np.floor((lon - lon_min) / CELL).astype(int)
    r = np.clip(r, 0, n_rows - 1)
    c = np.clip(c, 0, n_cols - 1)
    zone_id = r * n_cols + c
    return zone_id

def zone_id_to_rowcol_latlon(zone_id, n_rows, n_cols, lat_min, lon_min, CELL=0.001):
    """
    Convert a 1D zone index back into (row, col) and the corresponding (lat, lon) center.
    
    Parameters
    ----------
    zone_id : int or array-like
        Flattened 1D zone index (row * n_cols + col).
    n_rows : int
        Total number of rows in the grid.
    n_cols : int
        Total number of columns in the grid.
    lat_min : float
        Minimum latitude of the grid.
    lon_min : float
        Minimum longitude of the grid.
    CELL : float, default 0.001
        Cell size in degrees.
    
    Returns
    -------
    row : int or ndarray
        Row index (0-based, south → north).
    col : int or ndarray
        Column index (0-based, west → east).
    lat : float or ndarray
        Latitude of the cell center.
    lon : float or ndarray
        Longitude of the cell center.
    """
    zone_id = np.asarray(zone_id)

    row = zone_id // n_cols
    col = zone_id % n_cols

    # cell center = min + (index + 0.5)*CELL
    lat = lat_min + (row + 0.5) * CELL
    lon = lon_min + (col + 0.5) * CELL

    return row, col, lat, lon


def rle_lengths_by_value_1d(arr):
    """
    Run-length encode a 1D integer array and aggregate lengths by value.

    Parameters
    ----------
    arr : 1D array-like of int
        Sequence of zone indices for one agent over time.

    Returns
    -------
    out : dict[int, np.ndarray]
        Mapping: zone_id -> np.array of consecutive lengths (in steps).
        Example: [1,2,2,2,2,2,3,3,3,2] -> {1:[1], 2:[5,1], 3:[3]}
    """
    a = np.asarray(arr)
    if a.size == 0:
        return {}

    # Find starts of runs
    # mask[i] == True when i is the start of a run
    # (first position or value != previous)
    mask = np.empty(a.size, dtype=bool)
    mask[0] = True
    if a.size > 1:
        mask[1:] = a[1:] != a[:-1]

    run_starts = np.flatnonzero(mask)
    run_values = a[run_starts]
    run_lengths = np.diff(np.append(run_starts, a.size))

    # Aggregate lengths by value
    out = {}
    for v, L in zip(run_values, run_lengths):
        v = int(v)
        if v in out:
            out[v].append(int(L))
        else:
            out[v] = [int(L)]
    # Convert lists to compact arrays
    for k in out:
        out[k] = np.asarray(out[k], dtype=np.int32)
    return out

def compute_stay_lengths_by_agent(agent_location: pd.DataFrame) -> dict:
    """
    For each agent (row), compute consecutive stay lengths per zone.

    Parameters
    ----------
    agent_location : DataFrame (n_agents x n_steps)
        Each row corresponds to one agent; each column is a 5-minute time step.
        Cell values are integer zone indices.

    Returns
    -------
    result : dict
        Mapping: agent_key -> {zone_id: np.array of lengths}
        - agent_key is whatever is in agent_location.index (can be int/tuple/MultiIndex key).
    """
    values = agent_location.to_numpy()
    agents = agent_location.index
    result = {}

    for i, agent_key in enumerate(agents):
        result[agent_key] = rle_lengths_by_value_1d(values[i])

    return result



def haversine_m(
    lat1: float | np.ndarray | pd.Series,
    lon1: float | np.ndarray | pd.Series,
    lat2: float | np.ndarray | pd.Series,
    lon2: float | np.ndarray | pd.Series
) -> float | np.ndarray | pd.Series:
    """Vectorized haversine distance in meters.

    Parameters
    ----------
    lat1 : float | np.ndarray | pd.Series
    lon1 : float | np.ndarray | pd.Series
    lat2 : float | np.ndarray | pd.Series
    lon2 : float | np.ndarray | pd.Series

    Returns
    -------
    float | np.ndarray | pd.Series
        Great-circle distance(s) in meters.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))
def adjust_trajectory_single(df: pd.DataFrame, thresh_m: float = 30.0) -> pd.DataFrame:
    """
    For one agent's time-ordered trajectory:
    - Compute step distance.
    - If step <= thresh_m, treat as pseudo-move and copy previous *adjusted* point.
    - Else keep the original point.
    Requires columns: 'lat', 'lon'. If 'timestamp' exists, sorting uses it.
    """
    # sort if timestamp exists; otherwise keep current order
    if 'timestamp' in df.columns:
        g = df.sort_values('timestamp').copy()
    else:
        g = df.copy()

    lat = g['latitude'].to_numpy()
    lon = g['longitude'].to_numpy()

    # distances to previous original points (for decision)
    lat_prev = np.roll(lat, 1); lon_prev = np.roll(lon, 1)
    lat_prev[0] = lat[0]; lon_prev[0] = lon[0]
    step_m = haversine_m(lat_prev, lon_prev, lat, lon)
    step_m[0] = 0.0

    # sequential adjustment (depends on prior adjusted point)
    lat_adj = lat.copy()
    lon_adj = lon.copy()
    for i in range(1, len(g)):
        if step_m[i] <= thresh_m:
            # pseudo-move: hold last adjusted position
            lat_adj[i] = lat_adj[i-1]
            lon_adj[i] = lon_adj[i-1]
        # else keep original

    g['step_m']  = step_m
    g['lat_adj'] = lat_adj
    g['lon_adj'] = lon_adj
    return g

def hier_cluster_haversine(df: pd.DataFrame, lat_col: str ='lat_adj', lon_col: str ='lon_adj',
                           t_m: float =30.0, method:str ='single', drop_dups: bool =True):
    """
    Hierarchical clustering with haversine distance (meters) and cut at t_m (meters).
    - df: DataFrame with latitude/longitude columns (degrees).
    - lat_col/lon_col: column names to use (e.g., 'lat','lon' or 'lat_adj','lon_adj').
    - t_m: cut height in meters.
    - method: linkage method, e.g., 'single' (good for <=t connected components).
    - drop_dups: drop exact duplicate coords before clustering (faster).
    Returns: df_out with 'cluster_id' aligned to the original rows (duplicates share same label).
    """
    df_in = df[[lat_col, lon_col]].astype(float).copy()
    if drop_dups:
        uniq = df_in.drop_duplicates().reset_index()  # keep original indices
    else:
        uniq = df_in.reset_index()

    coords_deg = uniq[[lat_col, lon_col]].to_numpy()
    coords_rad = np.radians(coords_deg)

    # pairwise haversine distances (radians) -> meters
    dist_matrix_m = haversine_distances(coords_rad) * R
    dist_condensed_m = squareform(dist_matrix_m, checks=False)

    # hierarchical clustering
    Z = linkage(dist_condensed_m, method=method)
    labels = fcluster(Z, t=t_m, criterion='distance')

    # map labels back to original df
    lab_ser = pd.Series(labels, index=uniq['index'])
    df_out = df.copy()
    df_out['cluster_id'] = df_out.index.map(lab_ser)

    return df_out