#!/usr/bin/env python
# coding: utf-8

# In[1]:
''' 
    This file reads ONE agent location series, as clustered space zones and group-level forecasts as part of the features.
    Then it fits the individual-level DBCM model to learn space-time pattern of the agent, 
    inject possibly regime shift of some agents, calibrate Bayes factor thresholds of regime shift, 
    and finally determine if the agent indeed has a regime shift.    
'''

import Utilities2025.geo_utilities as mygeo
import Utilities2025.clustering_utilities as mycl
import Utilities2025.counting_utilities as mycount
import Utilities2025.run_model_utilities as myrun
import Utilities2025.staying_feature_utilities as mystay_x
import Utilities2025.anomaly_injection_utilities as myanomaly


import pandas as pd 
import numpy as np
import time
import pickle
from sklearn.metrics import accuracy_score
import os
import traceback

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


# agent_location_cl = pd.read_parquet(f"agent_location_cl_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.parquet", engine="pyarrow")
# agent_location_cl.columns = pd.to_datetime(agent_location_cl.columns)

# sampled_index = (
#     agent_location_cl.groupby(level=0, group_keys=False)
#       .apply(lambda g: g.sample(n=min(20, len(g)), replace=False, random_state=42))
#       .index
# )
# agent_location_cl_sub = agent_location_cl.loc[sampled_index,:]
# agent_location_cl_sub.to_parquet('agent_location_cl_sub.parquet')

agent_location_cl = pd.read_parquet('agent_location_cl_sub.parquet', engine="pyarrow")


def run_agent_full_analysis(
    agent_i_full: pd.Series,
    *,
    minimum_obs: int,
    end_of_trn: int,
    daily_len: int = 288,
    test_days: int = 3,
    start_cols_for_metrics: int = 288*7
):
    """
    End-to-end pipeline for one agent:
      - build transition indicators & zone types
      - compute cascade/group probabilities & features (p_hat, leave_gate)
      - fit PM (frozen after training) and OAM (continue from PM)
      - assemble multinomial predictions incl. rare transitions
      - compute accuracies and cumulative log Bayes factor
      - return the 'out' dict

    Requires globally available:
      - mystay_x: feature/injection helpers (compute_leave_gate_all, compute_empirical_transition_prob_all, kstep_trans_feature)
      - myrun:    modeling utilities (classify_zones_by_stay, get_transition_indicator, split_transitions_by_zone_type, etc.)
      - K_step_trans_mat: dict of transition cubes (e.g., {1: <SxS>, 2: ...})
    """
    
    # agent_i_full=agent_i_val
    # minimum_obs=minimum_obs
    # end_of_trn=val_st - 1
    # daily_len=288
    # test_days=3
    
    
    # 1) Split transitions by type
    zone_types = myrun.classify_zones_by_stay(agent_i_full.iloc[:end_of_trn])
    transition_counts_cascade, rare_directions_cascade = myrun.get_transition_indicator(
        agent_i_full,
        threshold=minimum_obs,
        end_of_trn=end_of_trn
    )
    directions_by_type = myrun.split_transitions_by_zone_type(transition_counts_cascade, zone_types)

    # 2) Group/cascade transition probabilities
    trans_prob_all = {
        k: mystay_x.kstep_trans_feature(k, agent_i_full, transition_counts_cascade, val[:,:,:len(agent_i_full)-k], min_zone_id=min_zone_id)
        for k, val in K_step_trans_mat.items()
    }
    cascade_prob_all = {
        k: (
            trans_prob_all[k]
            .groupby(level=0, group_keys=False)
            .apply(lambda g: myrun.multinomial_to_cascade(g))
        )
        for k in trans_prob_all.keys()
    }

    # 3) Features: leave_gate + empirical p_hat (OAM + PM/frozen)
    leave_gate_all = mystay_x.compute_leave_gate_all(
        directions_by_type=directions_by_type,
        daily_len=daily_len,
        test_days=test_days,
        end_of_trn=end_of_trn
    )
    p_hat_all = mystay_x.compute_empirical_transition_prob_all(
        directions_by_type,
        daily_len=daily_len,
        test_days=test_days,
        end_of_trn=end_of_trn,
        # Kernel-smoothed prob (staying_to_self)
        freq="5min",
        h_minutes=15,
        window_minutes_self=90,
        lambda_days=14,
        include_same_day=False,
        alpha0=0.5,
        beta0=0.5,
        # Switch prob (others)
        window_minutes_other=30,
        ewm_span=1
    )

    # 4) Restrict group trans probs to rows (ODs) present in each type
    group_trans_prob_all = {
        key: {k: val.loc[ind.index] for k, val in cascade_prob_all.items()}
        for key, ind in directions_by_type.items()
    }

    # 5) Feature spec per type
    feature_dict = {
        'staying_to_self':  {'feature': ['kernel_prob', 'lag1', 'lag2', 'leave_gate'], 'logit_feature': ['kernel_prob', 'lag1', 'lag2']},
        'staying_to_other': {'feature': ['kernel_prob', 'lag1', 'lag2'],                 'logit_feature': ['kernel_prob', 'lag1', 'lag2']},
        'passing_to_any':   {'feature': ['kernel_prob', 'lag1', 'lag2'],                 'logit_feature': ['kernel_prob', 'lag1', 'lag2']}
    }

    # 6) Fit PM and OAM across types
    results_by_type = myrun.run_oam_pm_pipeline(
        directions_input=directions_by_type,
        feature_dict=feature_dict,
        p_hat_all=p_hat_all,
        group_trans_prob_all=group_trans_prob_all,
        leave_gate_all=leave_gate_all,
        end_of_trn=end_of_trn,
        max_streak=1000,
        myrun=myrun
    )

    # 7) Assemble OAM multinomial, append rare, compute accuracy
    forecasted_prob_df_oam = pd.concat(
        [val['forecasted_prob_df_oam'] for val in results_by_type.values()],
        axis=0
    )
    multi_all_oam = myrun.assemble_multinomial_with_rare(
        forecasted_prob_df_oam,
        rare_directions_cascade,
        cascade_to_multinomial_func=myrun.cascade_to_multinomial
    )
    multi_all_oam = myrun.append_forecasted_rare_transitions(
        fitted_df=multi_all_oam,
        rare_cascade=rare_directions_cascade,
        trans_cube=K_step_trans_mat[1]
    )
    top_dest_oam = myrun.predict_argmax_dest_and_accuracy(
        multi_all_oam, agent_i_full, start_cols=start_cols_for_metrics
    )
    acc_oam = accuracy_score(top_dest_oam['real_dest'], top_dest_oam['dest'])
    OD_acc_oam = accuracy_score(
        top_dest_oam.loc[top_dest_oam['origin'] != top_dest_oam['real_dest'], 'real_dest'],
        top_dest_oam.loc[top_dest_oam['origin'] != top_dest_oam['real_dest'], 'dest']
    )

    # 8) Assemble PM multinomial, append rare, compute accuracy
    forecasted_prob_df_pm = pd.concat(
        [val['forecasted_prob_df_pm'] for val in results_by_type.values()],
        axis=0
    )
    multi_all_pm = myrun.assemble_multinomial_with_rare(
        forecasted_prob_df_pm,
        rare_directions_cascade,
        cascade_to_multinomial_func=myrun.cascade_to_multinomial
    )
    multi_all_pm = myrun.append_forecasted_rare_transitions(
        fitted_df=multi_all_pm,
        rare_cascade=rare_directions_cascade,
        trans_cube=K_step_trans_mat[1]
    )
    top_dest_pm = myrun.predict_argmax_dest_and_accuracy(
        multi_all_pm, agent_i_full, start_cols=start_cols_for_metrics
    )
    acc_pm = accuracy_score(top_dest_pm['real_dest'], top_dest_pm['dest'])
    OD_acc_pm = accuracy_score(
        top_dest_pm.loc[top_dest_pm['origin'] != top_dest_pm['real_dest'], 'real_dest'],
        top_dest_pm.loc[top_dest_pm['origin'] != top_dest_pm['real_dest'], 'dest']
    )

    # 9) Build combined summary + log Bayes factor
    idx = pd.MultiIndex.from_arrays(
        [agent_i_full.values[:-1], agent_i_full.values[1:]],
        names=['origin', 'dest']
    )
    # Align OAM/PM probabilities to actual next-step OD pairs
    prob_oam = pd.Series(
        multi_all_oam.values[multi_all_oam.index.get_indexer(idx), np.arange(0, multi_all_oam.shape[1])]
    )
    prob_pm = pd.Series(
        multi_all_pm.values[multi_all_oam.index.get_indexer(idx), np.arange(0, multi_all_oam.shape[1])]
    )

    combined_summary = pd.concat([idx.to_frame(index=False), prob_oam, prob_pm], axis=1)\
        .rename(columns={0: 'prob_oam', 1: 'prob_pm'})
    combined_summary['time'] = agent_i_full.index[1:]

    # log BF & run length
    log_probs = np.log(combined_summary[['prob_pm', 'prob_oam']].values)
    rowwise_diff = log_probs[:, 0] - log_probs[:, 1]
    log_L, run_length = myrun.log_cumulative_bayes_factor(rowwise_diff)
    combined_summary['log_L'] = log_L
    combined_summary['run_length'] = run_length

    # 10) Prepare final output dict
    # Use OAM summaries per type (concat with top-level keys)
    mod_analysis = pd.concat(
        [
            results_by_type['staying_to_self']['results_summary']['oam'],
            results_by_type['staying_to_other']['results_summary']['oam'],
            results_by_type['passing_to_any']['results_summary']['oam']
        ],
        axis=1,
        keys=['self', 'other', 'passing']
    )

    out = {
        'mod_analysis': mod_analysis,
        'path_acc': pd.Series({'Overall': acc_oam, 'Transition': OD_acc_oam}),
        'LogBF': combined_summary['log_L'].iloc[-1]
    }
    return out
    

# %%
    

with open(f"models_{threshold}_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.pkl", "rb") as f:
    models = pickle.load(f)

with open(f"fcasts_{threshold}_{int((CELL / 0.001)*100)}m_{int(radius_m)}_{int(max_size)}_{int(min_size)}.pkl", "rb") as f:
    fcasts = pickle.load(f)
    

fcasts_df = pd.DataFrame(
    np.vstack([val['p_hat'].values.reshape(1, -1) for key, val in fcasts.items()]),
    index=pd.MultiIndex.from_tuples(list(fcasts.keys()), names=["origin", "dest"]),
    columns=agent_location_cl.columns[1:]
)

fcast_multinom_df = myrun.stick_to_mult_by_group(fcasts_df, myrun.stick_to_mult_matrix)



one_step_trans_mat = mystay_x.build_full_transition(fcast_multinom_df)

st = time.time()
K_step_trans_mat = mystay_x.multistep_transition(one_step_trans_mat['transition_matrix'], K=2)
ed = time.time()
ed - st
K_step_trans_mat[1] = one_step_trans_mat['transition_matrix']
K_step_trans_mat = {k: K_step_trans_mat[k] for k in range(1, len(K_step_trans_mat)+1)}

# %%

'''
Pick an agent and then inject anomaly during the validation period. 
'''

test_days = 7; test_st = 21*288; test_end  = 28*288; daily_len = 288
minimum_obs = 5; min_zone_id = -121; nval = 50



taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])
agent_i = agent_location_cl.iloc[taskID,:]
agent_i.index = pd.to_datetime(agent_i. index)


res = myanomaly.inject_anomaly(agent_i, test_st, test_end)

if res['success']:
    agent_i_full = res['traj_new']
    # agent_i_full = agent_i
    end_of_trn = test_st - 1
    
    out_tst_regular = run_agent_full_analysis(
        agent_i_full=agent_i,
        minimum_obs=minimum_obs,
        end_of_trn=end_of_trn,
        daily_len=daily_len,
        test_days=test_days
    )
    
    out_tst_anomaly = run_agent_full_analysis(
        agent_i_full=agent_i_full,
        minimum_obs=minimum_obs,
        end_of_trn=end_of_trn,
        daily_len=daily_len,
        test_days=test_days
    )
    
    print([out_tst_regular['LogBF'],out_tst_anomaly['LogBF'] ])
    print(np.mean(res['traj_new'].iloc[test_st:test_end] != agent_i.iloc[test_st:test_end]))
    
    val_st = test_st - test_days * daily_len
    
    max_retry = 50
    st = time.time()
    all_val_lbf = np.zeros((nval, 3))
    for ii in range(nval):
        success = False
        for attempt in range(max_retry):
            try:
                # prepare validation trajectory
                if ii == 0:
                    agent_i_val = agent_i.iloc[:test_st]
                else:
                    agent_i_val, pairs = myanomaly.swap_days(
                        input_traj=agent_i.iloc[:test_st],
                        win_min=val_st,
                        win_max=test_st,
                        day_len=daily_len,
                        rng=None
                    )
    
                # inject anomaly
                res_val = myanomaly.inject_anomaly(agent_i_val, val_st, test_st)
                prop = np.mean(res_val['traj_new'].iloc[val_st:test_st] != agent_i_val.iloc[val_st:test_st])
                # skip to next iteration if injection failed
                if not res_val.get('success', True):
                    print(f"[ii={ii}] Anomaly injection failed. Skipping to next iteration.")
                    break
    
                # run analyses
                out_normal_val = run_agent_full_analysis(
                    agent_i_full=agent_i_val,
                    minimum_obs=minimum_obs,
                    end_of_trn=val_st - 1,
                    daily_len=daily_len,
                    test_days=test_days
                )
    
                out_anomaly_val = run_agent_full_analysis(
                    agent_i_full=res_val['traj_new'],
                    minimum_obs=minimum_obs,
                    end_of_trn=val_st - 1,
                    daily_len=daily_len,
                    test_days=test_days
                )
    
                all_val_lbf[ii, :] = [out_normal_val['LogBF'], \
                                      out_anomaly_val['LogBF'], \
                                      prop]
    
                success = True
                break  # exit retry loop if successful
    
            except Exception as e:
                print(f"[ii={ii}] Retry {attempt+1}/{max_retry} failed: {e}")
                traceback.print_exc()
    
        if not success:
            print(f"[ii={ii}] Failed after {max_retry} attempts. Skipping to next iteration.")
            continue
    
    ed = time.time()
    
    all_val_lbf = all_val_lbf[~np.any(all_val_lbf == 0, axis = 1)]
    best, curves = myanomaly.find_lbf_threshold2(np.sort(all_val_lbf[:,:2], axis=0)[::-1], plot = False, max_fpr = 0.1)
    
    output_results = {
        'out_tst_regular' :out_tst_regular,
        'out_tst_anomaly': out_tst_anomaly,
        'validation_logBF': all_val_lbf,
        'threshold': best,
        'curves': curves
        }
    with open(f"output_results_{taskID}.pkl", "wb") as f:
        pickle.dump(output_results, f)

