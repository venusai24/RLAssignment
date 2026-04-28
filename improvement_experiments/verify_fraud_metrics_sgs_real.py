"""
verify_fraud_metrics_sgs_real.py
================================
Runs the DDPG vs SGS comparison using parameters derived from creditcard.csv
(aligned with the paper's fraud detection case study).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper_implementation')))

import time
import numpy as np
import pandas as pd

from AlertDetectionEnv import AlertDetectionEnv
from DoubleOracleSolver import DoubleOracleSolver, PolicyContainer

 
from DDPG import DDPG_MIX_Oracle
from SGS import SGS_Oracle

 
 
 
ALERT_TYPES    = ['t1', 't2', 't3']
ATTACK_TYPES   = ['a1', 'a2', 'a3']
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

 
 
 
EPISODES_PER_ORACLE = 20     
MAX_DO_ITERATIONS   = 10     
EVAL_EPISODES       = 20     
MAX_STEPS           = 100    
SGS_N_STEPS         = 5     
SGS_N_BINS          = 3     

SCENARIOS = [
    {'label': 'B=10, D=2', 'B': 10, 'D': 2},
    {'label': 'B=50, D=5', 'B': 50, 'D': 5},
]


 
 
 

def create_env(B, D):
    P_funcs = {k: (lambda v=v: np.random.binomial(1, v)) for k, v in P_MATRIX.items()}
    F_funcs  = {t: (lambda m=m: np.random.poisson(m))
                for t, m in zip(ALERT_TYPES, FALSE_ALERT_MEANS)}
    return AlertDetectionEnv(
        alert_types=ALERT_TYPES, attack_types=ATTACK_TYPES,
        def_budget=B, att_budget=D,
        investigation_costs={t: 1.0 for t in ALERT_TYPES},
        attack_costs={ATTACK_TYPES[i]: ATTACKER_COSTS[i] for i in range(3)},
        attack_losses={ATTACK_TYPES[i]: DEFENDER_LOSSES[i] for i in range(3)},
        alert_probs=P_funcs, false_alert_probs=F_funcs,
    )


class MockEnvWrapper:
    def __init__(self, env):
        self.env   = env
        self.state = None

    def reset(self):
        self.state = self.env.reset()
        return self._flatten(self.state)

    def _flatten(self, state_tuple, role='adversary'):
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
                [v for a_dict in s.values() for v in a_dict.values()],
            ]).astype(np.float32)

    def step(self, def_action, att_action):
        alpha_plus = {t: min(int(def_action[i] * self.env.B), self.env.N[t])
                      for i, t in enumerate(self.env.T)}
        total_cost = sum(self.env.C[t] * alpha_plus[t] for t in self.env.T)
        if total_cost > self.env.B:
            scale = self.env.B / total_cost
            alpha_plus = {t: int(v * scale) for t, v in alpha_plus.items()}

        alpha_minus = {a: float(np.clip(att_action[i], 0.0, 1.0))
                       for i, a in enumerate(self.env.A)}
        att_cost = sum(self.env.E[a] * alpha_minus[a] for a in self.env.A)
        if att_cost > self.env.D:
            scale_att = self.env.D / att_cost
            alpha_minus = {a: v * scale_att for a, v in alpha_minus.items()}
            
        next_state, reward = self.env.step(alpha_plus, alpha_minus)
        self.state = next_state
        return next_state, reward, False, {}

    def get_observation(self, role):
        """Returns observation for a specific role."""
        return self._flatten(self.state, role)

 
 
 

def _safe_call(policy, env_wrapper, flat_state):
    import tensorflow as tf
    if isinstance(policy, tf.keras.Model):
        raw = policy(np.expand_dims(flat_state, 0), training=False)
        raw = raw.numpy()[0]
    elif hasattr(policy, '_best_action'):
        raw = policy._best_action(flat_state)
    else:
        raw = policy(flat_state)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 2: raw = raw[0]
    return raw

def evaluate_interaction(def_policy, att_policy, env_wrapper, episodes=EVAL_EPISODES, gamma=0.95):
    all_rewards = []
    for _ in range(episodes):
        env_wrapper.env.reset()
        env_wrapper.state = env_wrapper.env.get_state()
        ep_reward  = 0.0
        for k in range(MAX_STEPS):
            d_obs = env_wrapper.get_observation('defender')
            a_obs = env_wrapper.get_observation('adversary')
            
            d_act = def_policy(env_wrapper, d_obs)
            a_act = att_policy(env_wrapper, a_obs)
            _, reward, _, _ = env_wrapper.step(d_act, a_act)
            ep_reward += (gamma ** k) * reward
        all_rewards.append(ep_reward)
    return -float(np.mean(all_rewards))

def make_ddpg_oracle(role, B=None, D=None):
    return DDPG_MIX_Oracle(
        state_dim=DEF_STATE_DIM if role == 'defender' else ATT_STATE_DIM,
        action_dim=3,
        player_role=role,
        domain='fraud',
        action_costs=np.array(ATTACKER_COSTS) if role == 'adversary' else None,
        budget=D if role == 'adversary' else B,
    )

def make_sgs_oracle(role, B=None, D=None):
    return SGS_Oracle(
        state_dim=DEF_STATE_DIM if role == 'defender' else ATT_STATE_DIM,
        action_dim=3,
        player_role=role,
        n_steps=SGS_N_STEPS,
        n_bins=SGS_N_BINS,
        action_costs=np.array(ATTACKER_COSTS) if role == 'adversary' else None,
        budget=D if role == 'adversary' else B,
    )

class TimedDoubleOracleSolver(DoubleOracleSolver):
    def _wrap_policy(self, policy):
        """Ensures the policy returns a flat 1-D array and handles varying arguments."""
        def wrapped(*args):
             
            s = args[-1]
            w = args[0] if len(args) > 1 else None
            return _safe_call(policy, w, s)
        return wrapped

    def run(self, max_iterations=MAX_DO_ITERATIONS, episodes=EPISODES_PER_ORACLE):
        self.iterations_run = 0
        t0 = time.perf_counter()
        iteration = 0
        sigma_def = sigma_att = None
        while iteration < max_iterations:
            sigma_def, u_def_eq = self.solve_msne(self.container.utility_matrix)
            sigma_att, u_att_eq = self.solve_msne(-self.container.utility_matrix.T)
            new_pi_def = self.def_oracle.compute_best_response(self.env, self.container.att_policies, sigma_att, episodes=episodes, max_steps=MAX_STEPS)
            new_pi_att = self.att_oracle.compute_best_response(self.env, self.container.def_policies, sigma_def, episodes=episodes, max_steps=MAX_STEPS)
            u_new_def = self.evaluate_best_response_utility(new_pi_def, sigma_att, self.container.att_policies, is_defender=True)
            u_new_att = self.evaluate_best_response_utility(new_pi_att, sigma_def, self.container.def_policies, is_defender=False)
            def_improves = u_new_def > u_def_eq + 1e-5
            att_improves = u_new_att > u_att_eq + 1e-5
            iteration += 1
            self.iterations_run = iteration
            if not def_improves and not att_improves: break
            
             
            add_def = self._wrap_policy(new_pi_def) if def_improves else None
            add_att = self._wrap_policy(new_pi_att) if att_improves else None
            self.container.add_policies_and_update_matrix(add_def, add_att)
        self.elapsed = time.perf_counter() - t0
        
         
        sigma_def, _ = self.solve_msne(self.container.utility_matrix)
        sigma_att, _ = self.solve_msne(-self.container.utility_matrix.T)
        
        return sigma_def, sigma_att

def run_oracle_experiment(oracle_name, B, D, oracle_factory_def, oracle_factory_att):
    print(f"\nRunning {oracle_name} (B={B}, D={D})...")
    env = create_env(B, D)
    wrapper = MockEnvWrapper(env)
    def_oracle = oracle_factory_def(B=B, D=D)
    att_oracle = oracle_factory_att(B=B, D=D)
    
    def empirical_utility(dp, ap):
        wrapped_dp = lambda w, s: _safe_call(dp, w, s)
        wrapped_ap = lambda w, s: _safe_call(ap, w, s)
        return -evaluate_interaction(wrapped_dp, wrapped_ap, MockEnvWrapper(create_env(B, D)), episodes=2)

    init_def = [lambda s: np.ones(3) / 3]
    init_att = [lambda s: np.ones(3) / 3]
    container = PolicyContainer(init_def, init_att, empirical_utility)
    solver = TimedDoubleOracleSolver(def_oracle, att_oracle, container, wrapper)
    sigma_def, sigma_att = solver.run(max_iterations=MAX_DO_ITERATIONS, episodes=EPISODES_PER_ORACLE)
    
     
    def _def_mixed(env_w, s): return _safe_call(container.def_policies[np.random.choice(len(container.def_policies), p=sigma_def)], env_w, s)
    def _att_mixed(env_w, s): return _safe_call(container.att_policies[np.random.choice(len(container.att_policies), p=sigma_att)], env_w, s)
    arl_loss = evaluate_interaction(_def_mixed, _att_mixed, MockEnvWrapper(create_env(B, D)))
    
    return {'Oracle': oracle_name, 'Train_Time': solver.elapsed, 'ARL_Loss': arl_loss}

def run_comparison():
    results = []
    for sc in SCENARIOS:
        B, D = sc['B'], sc['D']
        results.append(run_oracle_experiment('DDPG-MIX', B, D, 
            lambda B, D: make_ddpg_oracle('defender', B, D), 
            lambda B, D: make_ddpg_oracle('adversary', B, D)))
        results.append(run_oracle_experiment('SGS (n-step Sarsa)', B, D, 
            lambda B, D: make_sgs_oracle('defender', B, D), 
            lambda B, D: make_sgs_oracle('adversary', B, D)))
    
    print("\n--- FINAL RESULTS (creditcard.csv - Paper Parameters) ---")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run_comparison()
