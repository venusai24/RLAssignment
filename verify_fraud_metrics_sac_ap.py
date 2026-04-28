"""
verify_fraud_metrics_sac_ap.py
===============================
Case Study II — Full verification pipeline (Figures 8–11).

Identical to verify_fraud_metrics.py EXCEPT:
  - Imports SAC_AP_Oracle instead of DDPG_MIX_Oracle
  - train_solver() instantiates SAC_AP_Oracle for both players

No existing files are modified.
"""

import os
import pickle
import hashlib
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import tensorflow as tf

from AlertDetectionEnv import AlertDetectionEnv
from DoubleOracleSolver import DoubleOracleSolver, PolicyContainer
from SAC_AP import SAC_AP_Oracle        


 
 
 

CHECKPOINT_FILE = "checkpoint_fraud_sac_ap.pkl"

def init_gpu():
    """Configures GPU memory growth to allow multiple processes to share VRAM."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass  

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(data, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return None

def get_task_id(func_name, task_params):
    """Generates a stable unique ID for a task."""
    s = f"{func_name}_{json.dumps(task_params, sort_keys=True)}"
    return hashlib.md5(s.encode()).hexdigest()

 
 
 

ALERT_TYPES       = ['t1', 't2', 't3']
ATTACK_TYPES      = ['a1', 'a2', 'a3']
FALSE_ALERT_MEANS = [10, 47, 39]
ATTACKER_COSTS    = [1, 3, 2]
DEFENDER_LOSSES   = [9.4, 12.1, 16.0]

P_MATRIX = {
    ('a1', 't1'): 0.9,  ('a1', 't2'): 0.61, ('a1', 't3'): 0.0,
    ('a2', 't1'): 0.09, ('a2', 't2'): 0.87, ('a2', 't3'): 0.12,
    ('a3', 't1'): 0.0,  ('a3', 't2'): 0.41, ('a3', 't3'): 0.85,
}


 
 
 

class MockEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.state = None

    def reset(self):
        self.state = self.env.reset()
        return self._flatten_state(self.state)

    def _flatten_state(self, state_tuple):
        n, m, s = state_tuple
        return np.concatenate([
            list(n.values()),
            list(m.values()),
            [val for a_dict in s.values() for val in a_dict.values()]
        ]).astype(np.float32)

    def step(self, def_action, att_action):
        env = self.env
        n_vals = self.state[0]   
        alpha_plus = {}
        for i, t in enumerate(env.T):
            alloc = def_action[i] * env.B
             
            alpha_plus[t] = min(alloc, n_vals[t])
        alpha_minus = {a: att_action[i] for i, a in enumerate(env.A)}

        next_state_tuple, reward = env.step(alpha_plus, alpha_minus)
        self.state = next_state_tuple
        return self._flatten_state(next_state_tuple), reward, False, {}

    def get_opponent_state(self):
        return self._flatten_state(self.state)


def dummy_utility(def_policy, att_policy):
    return 0.0


 
 
 

def defender_uniform(env_wrapper, flat_state):
    env = env_wrapper.env
    state = env_wrapper.state
    budget_per_type = env.B / len(env.T)
    raw = np.array([min(state[0][t], budget_per_type) / env.B for t in env.T])
    return raw


def defender_gain(env_wrapper, flat_state):
    env = env_wrapper.env
    state = env_wrapper.state
    priorities = {'t1': 9.4, 't2': 12.1, 't3': 16.0}
    sorted_t = sorted(env.T, key=lambda x: priorities[x], reverse=True)
    allocation = {t: 0.0 for t in env.T}
    rem_B = env.B
    for t in sorted_t:
        count = min(state[0][t], rem_B)
        allocation[t] = count
        rem_B -= count
    return np.array([allocation[t] / env.B for t in env.T])


def defender_rio(env_wrapper, flat_state):
    return defender_gain(env_wrapper, flat_state)


 
 
 

def attacker_uniform(env_wrapper, flat_state):
    env = env_wrapper.env
    weight = env.D / sum(env.E.values())
    return np.array([min(weight, 1.0) for _ in env.A])


def attacker_greedy(env_wrapper, flat_state):
    env = env_wrapper.env
    priorities = {a: env.L[a] * min(env.D / env.E[a], 1.0) for a in env.A}
    sorted_a = sorted(env.A, key=lambda x: priorities[x], reverse=True)
    allocation = {a: 0.0 for a in env.A}
    rem_D = env.D
    for a in sorted_a:
        if rem_D >= env.E[a]:
            allocation[a] = 1.0
            rem_D -= env.E[a]
        else:
            allocation[a] = rem_D / env.E[a]
            rem_D = 0
    return np.array([allocation[a] for a in env.A])


 
 
 

def evaluate_interaction(def_policy, att_policy, env_wrapper, episodes=10):
    gamma = 0.95
    all_rewards = []
    for _ in range(episodes):
        flat_state = env_wrapper.reset()
        episode_reward = 0.0
        for k in range(100):
            d_act = def_policy(env_wrapper, flat_state)
            if hasattr(d_act, 'numpy'): d_act = d_act.numpy()
            if d_act.ndim == 2:        d_act = d_act[0]

            a_act = att_policy(env_wrapper, flat_state)
            if hasattr(a_act, 'numpy'): a_act = a_act.numpy()
            if a_act.ndim == 2:        a_act = a_act[0]

            flat_state, reward, _, _ = env_wrapper.step(d_act, a_act)
            episode_reward += (gamma ** k) * reward
        all_rewards.append(episode_reward)
    return -np.mean(all_rewards)


def create_env(B, D):
    P_funcs = {k: (lambda v=v: np.random.binomial(1, v)) for k, v in P_MATRIX.items()}
    F_funcs = {t: (lambda m=m: np.random.poisson(m))
               for t, m in zip(ALERT_TYPES, FALSE_ALERT_MEANS)}
    return AlertDetectionEnv(
        alert_types=ALERT_TYPES, attack_types=ATTACK_TYPES,
        def_budget=B, att_budget=D,
        investigation_costs={t: 1.0 for t in ALERT_TYPES},
        attack_costs={ATTACK_TYPES[i]: ATTACKER_COSTS[i] for i in range(3)},
        attack_losses={ATTACK_TYPES[i]: DEFENDER_LOSSES[i] for i in range(3)},
        alert_probs=P_funcs, false_alert_probs=F_funcs
    )


 
 
 

def train_solver(B, D, max_iterations=15):
    env     = create_env(B, D)
    wrapper = MockEnvWrapper(env)
    state_dim = 3 + 3 + 9    

    def empirical_utility(dp, ap):
        _eval_env = MockEnvWrapper(create_env(B, D))
        wrapped_dp = lambda w, s: dp(s)
        wrapped_ap = lambda w, s: ap(s)
        return -evaluate_interaction(wrapped_dp, wrapped_ap, _eval_env, episodes=2)

     
    def_oracle = SAC_AP_Oracle(
        state_dim, action_dim=len(ALERT_TYPES),
        player_role='defender', domain='fraud', budget=B
    )
    att_oracle = SAC_AP_Oracle(
        state_dim, action_dim=len(ATTACK_TYPES),
        player_role='adversary', domain='fraud',
        action_costs=np.array(ATTACKER_COSTS), budget=D
    )
     

     
    def _safe_def_policy(s):
        """Uniform allocation clamped to alerts actually in the state."""
        n_vals = np.array(s[:len(ALERT_TYPES)], dtype=np.float32)    
        per_type = B / len(ALERT_TYPES)
        alloc = np.minimum(n_vals, per_type)
        return alloc / (B + 1e-8)    

    init_def_policies = [_safe_def_policy]
    def _safe_att_policy(s):
         
        costs = np.array(ATTACKER_COSTS)
        weight = D / (costs.sum() + 1e-8)
        return np.array([min(weight, 1.0) for _ in range(len(ATTACK_TYPES))])

    init_att_policies = [_safe_att_policy]

    container = PolicyContainer(init_def_policies, init_att_policies, empirical_utility)
    solver    = DoubleOracleSolver(def_oracle, att_oracle, container, wrapper)

    sigma_def, sigma_att = solver.run(max_iterations=max_iterations)
    return sigma_def, sigma_att, container


 
 
 

def worker_fig8(task):
    init_gpu()
    B, D, experiment = task['B'], task['D'], task['Experiment']
    sigma_def, sigma_att, container = train_solver(B, D, max_iterations=15)

    att_mixed = lambda w, s: container.att_policies[
        np.random.choice(len(container.att_policies), p=sigma_att)](s)
    def_mixed = lambda w, s: container.def_policies[
        np.random.choice(len(container.def_policies), p=sigma_def)](s)

    ew = MockEnvWrapper(create_env(B, D))
    res = {
        'Experiment': experiment, 'B': B, 'D': D,
        'SAC_AP':  evaluate_interaction(def_mixed,        att_mixed, ew, episodes=10),
        'Uniform': evaluate_interaction(defender_uniform,  att_mixed, ew, episodes=10),
        'GAIN':    evaluate_interaction(defender_gain,     att_mixed, ew, episodes=10),
        'RIO':     evaluate_interaction(defender_rio,      att_mixed, ew, episodes=10),
    }
    return ('fig8', res)


def worker_fig9(task):
    init_gpu()
    B, actual_D = task['B'], task['actual_D']

    sigma_def_est, _, cont_est = train_solver(B, 2, max_iterations=15)
    def_mixed_est = lambda w, s: cont_est.def_policies[
        np.random.choice(len(cont_est.def_policies), p=sigma_def_est)](s)

    _, sigma_att_act, cont_act = train_solver(B, actual_D, max_iterations=15)
    att_mixed_act = lambda w, s: cont_act.att_policies[
        np.random.choice(len(cont_act.att_policies), p=sigma_att_act)](s)

    ew = MockEnvWrapper(create_env(B, actual_D))
    res = {
        'Def_Budget': B, 'Actual_D': actual_D,
        'SAC_AP':  evaluate_interaction(def_mixed_est,   att_mixed_act, ew, episodes=10),
        'Uniform': evaluate_interaction(defender_uniform, att_mixed_act, ew, episodes=10),
        'GAIN':    evaluate_interaction(defender_gain,    att_mixed_act, ew, episodes=10),
        'RIO':     evaluate_interaction(defender_rio,     att_mixed_act, ew, episodes=10),
    }
    return ('fig9', res)


def worker_fig10(task):
    init_gpu()
    est_D, act_D, B = task['est_D'], task['act_D'], 20

    sigma_def_est, _, cont_est = train_solver(B, est_D, max_iterations=15)
    def_mixed_est = lambda w, s: cont_est.def_policies[
        np.random.choice(len(cont_est.def_policies), p=sigma_def_est)](s)

    _, sigma_att_act, cont_act = train_solver(B, act_D, max_iterations=15)
    att_mixed_act = lambda w, s: cont_act.att_policies[
        np.random.choice(len(cont_act.att_policies), p=sigma_att_act)](s)

    ew   = MockEnvWrapper(create_env(B, act_D))
    loss = evaluate_interaction(def_mixed_est, att_mixed_act, ew, episodes=10)
    return ('fig10', {'est_D': est_D, 'act_D': act_D, 'Loss': loss})


def worker_fig11(task):
    init_gpu()
    B, D = task['B'], 2

    sigma_def, sigma_att, container = train_solver(B, D, max_iterations=15)
    att_mixed = lambda w, s: container.att_policies[
        np.random.choice(len(container.att_policies), p=sigma_att)](s)
    def_mixed = lambda w, s: container.def_policies[
        np.random.choice(len(container.def_policies), p=sigma_def)](s)

    att_policies = {'Greedy': attacker_greedy, 'Uniform': attacker_uniform, 'SAC_AP': att_mixed}
    def_policies = {'SAC_AP': def_mixed, 'Uniform': defender_uniform,
                    'GAIN': defender_gain, 'RIO': defender_rio}

    ew = MockEnvWrapper(create_env(B, D))
    res_list = []
    for a_name, a_func in att_policies.items():
        for d_name, d_func in def_policies.items():
            loss = evaluate_interaction(d_func, a_func, ew, episodes=10)
            res_list.append({'B': B, 'Attacker': a_name, 'Defender': d_name, 'Loss': loss})
    return ('fig11', res_list)


 
 
 

def run_full_verification():
    tasks_config = []

    for B in [10, 20, 30]:
        tasks_config.append((worker_fig8,  {'Experiment': 'Fig 8 Left',  'B': B,  'D': 2}))
    for D in [1, 2, 3]:
        tasks_config.append((worker_fig8,  {'Experiment': 'Fig 8 Right', 'B': 20, 'D': D}))

    for B in [10, 30]:
        for D in [1, 2, 3]:
            tasks_config.append((worker_fig9,  {'B': B, 'actual_D': D}))

    for est_D in [1, 2, 3]:
        for act_D in [1, 2, 3]:
            tasks_config.append((worker_fig10, {'est_D': est_D, 'act_D': act_D}))

    for B in [10, 30]:
        tasks_config.append((worker_fig11, {'B': B}))

     
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint: {len(checkpoint['completed_ids'])} tasks already done.")
        fig8_results  = checkpoint['fig8_results']
        fig9_results  = checkpoint['fig9_results']
        fig10_matrix  = checkpoint['fig10_matrix']
        fig11_results = checkpoint['fig11_results']
        completed_ids = checkpoint['completed_ids']
    else:
        fig8_results  = []
        fig9_results  = []
        fig10_matrix  = np.zeros((3, 3))
        fig11_results = []
        completed_ids = set()

     
    pending_tasks = []
    for func, arg in tasks_config:
        tid = get_task_id(func.__name__, arg)
        if tid not in completed_ids:
            pending_tasks.append((func, arg, tid))

    if not pending_tasks:
        print("All tasks are already completed.")
    else:
        print(f"Submitting {len(pending_tasks)} pending tasks to ProcessPoolExecutor (max_workers=6) ...")
         
        with ProcessPoolExecutor(max_workers=6) as executor:
            future_to_tid = {executor.submit(func, arg): tid for func, arg, tid in pending_tasks}
            
            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                try:
                    task_type, res = future.result()
                    
                    if task_type == 'fig8':
                        fig8_results.append(res)
                        print(f"  Fig 8 done: B={res['B']}, D={res['D']}")
                    elif task_type == 'fig9':
                        fig9_results.append(res)
                        print(f"  Fig 9 done: Def_B={res['Def_Budget']}, Act_D={res['Actual_D']}")
                    elif task_type == 'fig10':
                        fig10_matrix[res['act_D']-1, res['est_D']-1] = res['Loss']
                        print(f"  Fig10 done: est_D={res['est_D']}, act_D={res['act_D']}")
                    elif task_type == 'fig11':
                        fig11_results.extend(res)
                        print("  Fig11 chunk done")
                    
                    completed_ids.add(tid)
                     
                    save_checkpoint({
                        'fig8_results': fig8_results,
                        'fig9_results': fig9_results,
                        'fig10_matrix': fig10_matrix,
                        'fig11_results': fig11_results,
                        'completed_ids': completed_ids
                    })
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")

    print("\n" + "="*55)
    print("--- Figure 8  (Full Knowledge Baseline — SAC-AP) ---")
    df8 = pd.DataFrame(fig8_results)
    print(df8.to_string(index=False))

    print("\n" + "="*55)
    print("--- Figure 9  (Uncertain Budget — SAC-AP) ---")
    df9 = pd.DataFrame(fig9_results)
    print(df9.to_string(index=False))

    print("\n" + "="*55)
    print("--- Figure 10 (Loss Matrix — SAC-AP) ---")
    print("Columns: Estimated D [1,2,3]  |  Rows: Actual D [1,2,3]")
    print(pd.DataFrame(fig10_matrix, index=[1,2,3], columns=[1,2,3]))

    print("\n" + "="*55)
    print("--- Figure 11 (Uncertain Attack Policy — SAC-AP) ---")
    df11 = pd.DataFrame(fig11_results)
    print("\nFigure 11  B=10:")
    print(df11[df11['B'] == 10].pivot(
        index='Attacker', columns='Defender', values='Loss'
    ).loc[['Greedy','Uniform','SAC_AP'], ['SAC_AP','Uniform','GAIN','RIO']])
    print("\nFigure 11  B=30:")
    print(df11[df11['B'] == 30].pivot(
        index='Attacker', columns='Defender', values='Loss'
    ).loc[['Greedy','Uniform','SAC_AP'], ['SAC_AP','Uniform','GAIN','RIO']])


if __name__ == "__main__":
    run_full_verification()
