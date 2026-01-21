# Bayesian Modeling and Monitoring of Large-Scale Agent Flows in Dynamic Networks

This repository contains the code used for the PhD thesis chapter:

**“Bayesian Modeling and Monitoring of Large-Scale Agent Flows in Dynamic Networks.”**

The code implements a Bayesian modeling pipeline for large-scale agent mobility data on dynamic networks, including data cleaning, spatial discretization and clustering, group-level flow modeling, and individual-level regime-shift detection using dynamic Bayesian models.

---

## Repository Structure

### `clean_knoxville.py`

This script preprocesses the raw Knoxville mobility dataset and constructs spatial zones used throughout the modeling pipeline. The main steps include:

1. **Noise correction of raw GPS coordinates**  
   For 5-minute resolution data, if the Haversine distance between consecutive locations is less than 50 meters, the movement is treated as a spurious transition and the new location is forced to equal the previous one.

2. **Spatial discretization**  
   The cleaned latitude/longitude coordinates are discretized into fixed spatial cells with side length 500 meters.

3. **Spatial clustering of zones**  
   Spatial clustering is performed using an adjacency matrix constructed via a BallTree, where nodes within a distance threshold are connected.  
   Breadth-first search (BFS) is applied to refine clusters and enforce a maximum connected-component size constraint.

Clustering is conducted separately for **passing zones** and **staying zones**. After the initial BallTree-based clustering, small clusters are merged or reclustered to avoid an excessive number of tiny, isolated clusters.

---

### `run_group_model.py`

This script fits the **group-level mobility model**.

It reads agent trajectories represented as sequences of discrete spatial zone indices, estimates aggregate flow dynamics at the group level, and generates group-level forecasts. These forecasts are later used as informative features in the individual-level models.

---

### `run_detection_sim.py`

This script implements the **individual-level regime-shift detection model**.

For a single agent, it:
- Reads the agent’s trajectory in clustered spatial zones
- Incorporates group-level forecasts as covariates
- Fits an individual-level Dynamic Bayesian Cascade Model (DBCM) to learn space–time mobility patterns
- Injects synthetic regime shifts for calibration
- Calibrates Bayes factor thresholds for detecting regime changes
- Determines whether the agent exhibits a statistically significant regime shift

---

## Notes

- The code is designed for research and reproducibility purposes.
- Outputs from earlier stages (e.g., clustering and group-level forecasts) are assumed to be saved and reused by downstream scripts.
- This repository corresponds specifically to one chapter of the PhD thesis and is not intended as a general-purpose software package.

---

## Citation

If you use or reference this code, please cite the corresponding PhD thesis chapter.
