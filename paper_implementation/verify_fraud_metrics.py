import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from AlertDetectionEnv import AlertDetectionEnv
from DDPG import DDPG_MIX_Oracle
from DoubleOracleSolver import DoubleOracleSolver, PolicyContainer

 
ALERT_TYPES = ['t1', 't2', 't3']
ATTACK_TYPES = ['a1', 'a2', 'a3']
FALSE_ALERT_MEANS = [10, 47, 39]
ATTACKER_COSTS = [1, 3, 2]
DEFENDER_LOSSES = [9.4, 12.1, 16.0]

P_MATRIX = {
    ('a1', 't1'): 0.9, ('a1', 't2'): 0.61, ('a1', 't3'): 0.0,
    ('a2', 't1'): 0.09, ('a2', 't2'): 0.87, ('a2', 't3'): 0.12,
    ('a3', 't1'): 0.0, ('a3', 't2'): 0.41, ('a3', 't3'): 0.85
}

# State Dimensions aligned with paper (Section III-D and Table II)
DEF_STATE_DIM = 3   # |T| (counts of alerts of each type)
ATT_STATE_DIM = 15  # |T| + |A| * (1 + |T|) (N, M, and S)

class MockEnvWrapper:
    """Wrapper to interact with DDPG Oracle and existing Env"""
    def __init__(self, env):
        self.env = env
        self.state = None
        
    def reset(self):
        self.state = self.env.reset()
        return self._flatten_state(self.state)
        
    def _flatten_state(self, state_tuple, role='adversary'):
        """Flattens state tuple according to the player's knowledge."""
        n, m, s = state_tuple
        if role == 'defender':
            # Defender only observes counts of uninvestigated alerts (N)
            return np.array(list(n.values())).astype(np.float32)
        else:
            # Adversary observes full state (N, M, S)
            return np.concatenate([
                list(n.values()),
                list(m.values()),
                [val for a_dict in s.values() for val in a_dict.values()]
            ]).astype(np.float32)

    def step(self, def_action, att_action):
        alpha_plus = {t: def_action[i] * self.env.B for i, t in enumerate(self.env.T)}
        alpha_minus = {a: att_action[i] for i, a in enumerate(self.env.A)}
        
        next_state_tuple, reward = self.env.step(alpha_plus, alpha_minus)
        self.state = next_state_tuple
        return next_state_tuple, reward, False, {}

    def get_observation(self, role):
        """Returns observation for a specific role."""
        return self._flatten_state(self.state, role)

def dummy_utility(def_policy, att_policy):
    return 0.0

 
def defender_uniform(env_wrapper, flat_state):
    env = env_wrapper.env
    state = env_wrapper.state
    budget_per_type = env.B / len(env.T)
    raw_action = np.array([min(state[0][t], budget_per_type) / env.B for t in env.T])
    return raw_action

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
    raw_action = np.array([allocation[t] / env.B for t in env.T])
    return raw_action

def defender_rio(env_wrapper, flat_state):
    return defender_gain(env_wrapper, flat_state)

 
def attacker_uniform(env_wrapper, flat_state):
    env = env_wrapper.env
    D = env.D
    weight = D / sum(env.E.values())
    raw_action = np.array([min(weight, 1.0) for a in env.A])
    return raw_action

def attacker_greedy(env_wrapper, flat_state):
    env = env_wrapper.env
    D = env.D
    priorities = {a: env.L[a] * min(D / env.E[a], 1.0) for a in env.A}
    sorted_a = sorted(env.A, key=lambda x: priorities[x], reverse=True)
    allocation = {a: 0.0 for a in env.A}
    rem_D = D
    for a in sorted_a:
        if rem_D >= env.E[a]:
            allocation[a] = 1.0
            rem_D -= env.E[a]
        else:
            allocation[a] = rem_D / env.E[a]
            rem_D = 0
    raw_action = np.array([allocation[a] for a in env.A])
    return raw_action

 
def evaluate_interaction(def_policy, att_policy, env_wrapper, episodes=10):
    gamma = 0.95
    all_rewards = []
    for _ in range(episodes):
        env_wrapper.env.reset()
        env_wrapper.state = env_wrapper.env.get_state()
        episode_reward = 0
        for k in range(50):
            d_obs = env_wrapper.get_observation('defender')
            a_obs = env_wrapper.get_observation('adversary')
            
            d_act_raw = def_policy(env_wrapper, d_obs)
            if hasattr(d_act_raw, 'numpy'): d_act_raw = d_act_raw.numpy()
            if d_act_raw.ndim == 2: d_act_raw = d_act_raw[0]
            
            a_act_raw = att_policy(env_wrapper, a_obs)
            if hasattr(a_act_raw, 'numpy'): a_act_raw = a_act_raw.numpy()
            if a_act_raw.ndim == 2: a_act_raw = a_act_raw[0]
            
            _, reward, _, _ = env_wrapper.step(d_act_raw, a_act_raw)
            episode_reward += (gamma ** k) * reward
        all_rewards.append(episode_reward)
    return -np.mean(all_rewards)

def create_env(B, D):
    P_funcs = {}
    for k, v in P_MATRIX.items():
        P_funcs[k] = lambda v_val=v: np.random.binomial(1, v_val)
    F_funcs = {}
    for i, t in enumerate(ALERT_TYPES):
        mean = FALSE_ALERT_MEANS[i]
        F_funcs[t] = lambda m=mean: np.random.poisson(m)
        
    return AlertDetectionEnv(
        alert_types=ALERT_TYPES, attack_types=ATTACK_TYPES,
        def_budget=B, att_budget=D,
        investigation_costs={t: 1.0 for t in ALERT_TYPES},
        attack_costs={ATTACK_TYPES[i]: ATTACKER_COSTS[i] for i in range(3)},
        attack_losses={ATTACK_TYPES[i]: DEFENDER_LOSSES[i] for i in range(3)},
        alert_probs=P_funcs, false_alert_probs=F_funcs
    )

def train_solver(B, D, max_iterations=2):
    env = create_env(B, D)
    wrapper = MockEnvWrapper(env)
    state_dim = 3 + 3 + 9
    
    def empirical_utility(dp, ap):
        wrapped_dp = lambda w, s: dp(s)
        wrapped_ap = lambda w, s: ap(s)
        return -evaluate_interaction(wrapped_dp, wrapped_ap, MockEnvWrapper(create_env(B, D)), episodes=2)
        
    def_oracle = DDPG_MIX_Oracle(DEF_STATE_DIM, action_dim=len(ALERT_TYPES), player_role='defender', domain='fraud', budget=B)
    att_oracle = DDPG_MIX_Oracle(ATT_STATE_DIM, action_dim=len(ATTACK_TYPES), player_role='adversary', domain='fraud', action_costs=np.array(ATTACKER_COSTS), budget=D)
    
    init_def_policies = [lambda s: np.ones(len(ALERT_TYPES)) / len(ALERT_TYPES)]
    init_att_policies = [lambda s: np.ones(len(ATTACK_TYPES)) / len(ATTACK_TYPES)]
    
    container = PolicyContainer(init_def_policies, init_att_policies, empirical_utility)
    solver = DoubleOracleSolver(def_oracle, att_oracle, container, wrapper)
    
    sigma_def, sigma_att = solver.run(max_iterations=max_iterations)
    return sigma_def, sigma_att, container

def worker_fig8(task):
    B = task['B']
    D = task['D']
    experiment = task['Experiment']
    sigma_def, sigma_att, container = train_solver(B, D, max_iterations=2)
    
    att_mixed = lambda w, s: container.att_policies[np.random.choice(len(container.att_policies), p=sigma_att)](s)
    def_mixed = lambda w, s: container.def_policies[np.random.choice(len(container.def_policies), p=sigma_def)](s)
    
    eval_wrapper = MockEnvWrapper(create_env(B, D))
    res = {
        'Experiment': experiment, 'B': B, 'D': D,
        'ARL': evaluate_interaction(def_mixed, att_mixed, eval_wrapper, episodes=10),
        'Uniform': evaluate_interaction(defender_uniform, att_mixed, eval_wrapper, episodes=10),
        'GAIN': evaluate_interaction(defender_gain, att_mixed, eval_wrapper, episodes=10),
        'RIO': evaluate_interaction(defender_rio, att_mixed, eval_wrapper, episodes=10)
    }
    return ('fig8', res)

def worker_fig9(task):
    B = task['B']
    actual_D = task['actual_D']
    
     
    sigma_def_est, _, cont_est = train_solver(B, 2, max_iterations=2)
    def_mixed_est = lambda w, s: cont_est.def_policies[np.random.choice(len(cont_est.def_policies), p=sigma_def_est)](s)
    
     
    _, sigma_att_act, cont_act = train_solver(B, actual_D, max_iterations=2)
    att_mixed_act = lambda w, s: cont_act.att_policies[np.random.choice(len(cont_act.att_policies), p=sigma_att_act)](s)
    
    eval_wrapper = MockEnvWrapper(create_env(B, actual_D))
    res = {
        'Def_Budget': B, 'Actual_D': actual_D,
        'ARL': evaluate_interaction(def_mixed_est, att_mixed_act, eval_wrapper, episodes=10),
        'Uniform': evaluate_interaction(defender_uniform, att_mixed_act, eval_wrapper, episodes=10),
        'GAIN': evaluate_interaction(defender_gain, att_mixed_act, eval_wrapper, episodes=10),
        'RIO': evaluate_interaction(defender_rio, att_mixed_act, eval_wrapper, episodes=10)
    }
    return ('fig9', res)

def worker_fig10(task):
    est_D = task['est_D']
    act_D = task['act_D']
    B = 20
    
    sigma_def_est, _, cont_est = train_solver(B, est_D, max_iterations=2)
    def_mixed_est = lambda w, s: cont_est.def_policies[np.random.choice(len(cont_est.def_policies), p=sigma_def_est)](s)
    
    _, sigma_att_act, cont_act = train_solver(B, act_D, max_iterations=2)
    att_mixed_act = lambda w, s: cont_act.att_policies[np.random.choice(len(cont_act.att_policies), p=sigma_att_act)](s)
    
    eval_wrapper = MockEnvWrapper(create_env(B, act_D))
    loss = evaluate_interaction(def_mixed_est, att_mixed_act, eval_wrapper, episodes=10)
    
    return ('fig10', {'est_D': est_D, 'act_D': act_D, 'Loss': loss})

def worker_fig11(task):
    B = task['B']
    D = 2
    
    sigma_def, sigma_att, container = train_solver(B, D, max_iterations=2)
    
    att_mixed = lambda w, s: container.att_policies[np.random.choice(len(container.att_policies), p=sigma_att)](s)
    def_mixed = lambda w, s: container.def_policies[np.random.choice(len(container.def_policies), p=sigma_def)](s)
    
    att_policies = {'Greedy': attacker_greedy, 'Uniform': attacker_uniform, 'ARL': att_mixed}
    def_policies = {'ARL': def_mixed, 'Uniform': defender_uniform, 'GAIN': defender_gain, 'RIO': defender_rio}
    
    eval_wrapper = MockEnvWrapper(create_env(B, D))
    res_list = []
    for a_name, a_func in att_policies.items():
        for d_name, d_func in def_policies.items():
            loss = evaluate_interaction(d_func, a_func, eval_wrapper, episodes=10)
            res_list.append({'B': B, 'Attacker': a_name, 'Defender': d_name, 'Loss': loss})
            
    return ('fig11', res_list)

def run_full_verification():
    tasks = []
    
    for B in [10, 20, 30]:
        tasks.append((worker_fig8, {'Experiment': 'Fig 8 Left', 'B': B, 'D': 2}))
    for D in [1, 2, 3]:
        tasks.append((worker_fig8, {'Experiment': 'Fig 8 Right', 'B': 20, 'D': D}))
        
    for B in [10, 30]:
        for D in [1, 2, 3]:
            tasks.append((worker_fig9, {'B': B, 'actual_D': D}))

    for est_D in [1, 2, 3]:
        for act_D in [1, 2, 3]:
            tasks.append((worker_fig10, {'est_D': est_D, 'act_D': act_D}))

    for B in [10, 30]:
        tasks.append((worker_fig11, {'B': B}))

    fig8_results = []
    fig9_results = []
    fig10_matrix = np.zeros((3, 3))
    fig11_results = []

    print("Submitting tasks to ProcessPoolExecutor for parallel execution...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, arg) for func, arg in tasks]
        for future in futures:
            task_type, res = future.result()
            if task_type == 'fig8':
                fig8_results.append(res)
                print(f"Completed Fig 8: B={res['B']}, D={res['D']}")
            elif task_type == 'fig9':
                fig9_results.append(res)
                print(f"Completed Fig 9: Def_B={res['Def_Budget']}, Act_D={res['Actual_D']}")
            elif task_type == 'fig10':
                 
                fig10_matrix[res['act_D']-1, res['est_D']-1] = res['Loss']
                print(f"Completed Fig 10: Est_D={res['est_D']}, Act_D={res['act_D']}")
            elif task_type == 'fig11':
                fig11_results.extend(res)
                print(f"Completed Fig 11 chunk")

    print("\n" + "="*50)
    print("--- Verifying Figure 8 (Full Knowledge Baseline)")
    df_fig8 = pd.DataFrame(fig8_results)
    print(df_fig8.to_string(index=False))

    print("\n" + "="*50)
    print("--- Verifying Figure 9 (Uncertain Budget) ---")
    df_fig9 = pd.DataFrame(fig9_results)
    print(df_fig9.to_string(index=False))

    print("\n" + "="*50)
    print("--- Verifying Figure 10 (Loss Matrix) ---")
    print("Columns: Estimated D [1, 2, 3] | Rows: Actual D [1, 2, 3]")
    print(pd.DataFrame(fig10_matrix, index=[1,2,3], columns=[1,2,3]))

    print("\n" + "="*50)
    print("--- Verifying Figure 11 (Uncertain Attack Policy) ---")
    df11 = pd.DataFrame(fig11_results)
    print("\nFigure 11 Matrix for B=10:")
    print(df11[df11['B'] == 10].pivot(index='Attacker', columns='Defender', values='Loss').loc[['Greedy', 'Uniform', 'ARL'], ['ARL', 'Uniform', 'GAIN', 'RIO']])
    print("\nFigure 11 Matrix for B=30:")
    print(df11[df11['B'] == 30].pivot(index='Attacker', columns='Defender', values='Loss').loc[['Greedy', 'Uniform', 'ARL'], ['ARL', 'Uniform', 'GAIN', 'RIO']])

if __name__ == "__main__":
    run_full_verification()