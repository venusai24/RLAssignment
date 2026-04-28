"""Comparison between DDPG-MIX and SGS Oracle on Credit Card Fraud Dataset."""

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
    ('a1', 't1'): 0.9,  ('a1', 't2'): 0.61, ('a1', 't3'): 0.0,
    ('a2', 't1'): 0.09, ('a2', 't2'): 0.87, ('a2', 't3'): 0.12,
    ('a3', 't1'): 0.0,  ('a3', 't2'): 0.41, ('a3', 't3'): 0.85,
}

STATE_DIM  = 3 + 3 + 9    

 
 
 
EPISODES_PER_ORACLE = 50     
MAX_DO_ITERATIONS   = 2      
EVAL_EPISODES       = 10     
MAX_STEPS           = 50     
SGS_N_STEPS         = 5      
SGS_N_BINS          = 3      

SCENARIOS = [
    {'label': 'B=10, D=2', 'B': 10, 'D': 2},
    {'label': 'B=20, D=2', 'B': 20, 'D': 2},
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
    """Wraps AlertDetectionEnv for oracle interaction."""

    def __init__(self, env):
        self.env   = env
        self.state = None

    def reset(self):
        self.state = self.env.reset()
        return self._flatten(self.state)

    def _flatten(self, state_tuple):
        n, m, s = state_tuple
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
        return self._flatten(next_state), reward, False, {}

    def get_opponent_state(self):
        return self._flatten(self.state)


 
 
 

def defender_uniform(env_wrapper, flat_state):
    env = env_wrapper.env
    budget_per_type = env.B / len(env.T)
    raw = np.array([
        min(env_wrapper.state[0][t], budget_per_type) / env.B for t in env.T
    ])
    return raw

def defender_gain(env_wrapper, flat_state):
    env = env_wrapper.env
    priorities = {'t1': 9.4, 't2': 12.1, 't3': 16.0}
    sorted_t = sorted(env.T, key=lambda x: priorities[x], reverse=True)
    allocation = {t: 0.0 for t in env.T}
    rem_B = env.B
    for t in sorted_t:
        count = min(env_wrapper.state[0][t], rem_B)
        allocation[t] = count
        rem_B -= count
    return np.array([allocation[t] / env.B for t in env.T])

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
            allocation[a] = 1.0; rem_D -= env.E[a]
        else:
            allocation[a] = rem_D / env.E[a]; rem_D = 0
    return np.array([allocation[a] for a in env.A])


 
 
 

def evaluate_interaction(def_policy, att_policy, env_wrapper,
                         episodes=EVAL_EPISODES, gamma=0.95):
    """Returns mean discounted defender loss (positive = worse)."""
    all_rewards = []
    for _ in range(episodes):
        flat_state = env_wrapper.reset()
        ep_reward  = 0.0
        for k in range(MAX_STEPS):
            d_act = def_policy(env_wrapper, flat_state)
            if hasattr(d_act, 'numpy'): d_act = d_act.numpy()
            if d_act.ndim == 2:         d_act = d_act[0]

            a_act = att_policy(env_wrapper, flat_state)
            if hasattr(a_act, 'numpy'): a_act = a_act.numpy()
            if a_act.ndim == 2:         a_act = a_act[0]

            flat_state, reward, _, _ = env_wrapper.step(d_act, a_act)
            ep_reward += (gamma ** k) * reward
        all_rewards.append(ep_reward)
    return -float(np.mean(all_rewards))    


 
 
 

def make_ddpg_oracle(role, B=None, D=None):
    """Constructs a DDPG_MIX_Oracle with fraud-domain settings."""
    return DDPG_MIX_Oracle(
        state_dim=STATE_DIM,
        action_dim=len(ALERT_TYPES) if role == 'defender' else len(ATTACK_TYPES),
        player_role=role,
        domain='fraud',
        action_costs=np.array(ATTACKER_COSTS) if role == 'adversary' else None,
        budget=D if role == 'adversary' else B,
    )

def make_sgs_oracle(role, B=None, D=None):
    """Constructs an SGS_Oracle with matching settings."""
    return SGS_Oracle(
        state_dim=STATE_DIM,
        action_dim=len(ALERT_TYPES) if role == 'defender' else len(ATTACK_TYPES),
        player_role=role,
        n_steps=SGS_N_STEPS,
        n_bins=SGS_N_BINS,
        action_costs=np.array(ATTACKER_COSTS) if role == 'adversary' else None,
        budget=D if role == 'adversary' else B,
    )


 
 
 

class TimedDoubleOracleSolver(DoubleOracleSolver):
    """Subclass that records iteration count and total training time."""

    def run(self, max_iterations=MAX_DO_ITERATIONS, episodes=EPISODES_PER_ORACLE):
        """Runs the Double Oracle loop with recorded metrics."""
        self.iterations_run = 0
        t0 = time.perf_counter()

        iteration = 0
        sigma_def = sigma_att = None

        while iteration < max_iterations:
            sigma_def, u_def_eq = self.solve_msne(self.container.utility_matrix)
            sigma_att, u_att_eq = self.solve_msne(-self.container.utility_matrix.T)

            new_pi_def = self.def_oracle.compute_best_response(
                self.env, self.container.att_policies, sigma_att,
                episodes=episodes, max_steps=MAX_STEPS,
            )
            new_pi_att = self.att_oracle.compute_best_response(
                self.env, self.container.def_policies, sigma_def,
                episodes=episodes, max_steps=MAX_STEPS,
            )

            u_new_def = self.evaluate_best_response_utility(
                new_pi_def, sigma_att, self.container.att_policies, is_defender=True
            )
            u_new_att = self.evaluate_best_response_utility(
                new_pi_att, sigma_def, self.container.def_policies, is_defender=False
            )

            def_improves = u_new_def > u_def_eq + 1e-5
            att_improves = u_new_att > u_att_eq + 1e-5

            iteration += 1
            self.iterations_run = iteration

            if not def_improves and not att_improves:
                print(f"Converged at iteration {iteration}.")
                break

            add_def = new_pi_def if def_improves else None
            add_att = new_pi_att if att_improves else None
            self.container.add_policies_and_update_matrix(add_def, add_att)

        self.elapsed = time.perf_counter() - t0
        return sigma_def, sigma_att


 
 
 

def run_oracle_experiment(oracle_name, B, D, oracle_factory_def, oracle_factory_att):
    """Runs a single (oracle, B, D) experiment configuration."""
    print(f"  Oracle: {oracle_name} | B={B}, D={D}")

    env    = create_env(B, D)
    wrapper = MockEnvWrapper(env)

    def_oracle = oracle_factory_def(B=B, D=D)
    att_oracle = oracle_factory_att(B=B, D=D)

     
    def _safe_call(policy, env_wrapper, flat_state):
        """Unified policy calling convention for functions, oracles, and models."""
        import tensorflow as tf
        if isinstance(policy, tf.keras.Model):
             
            raw = policy(np.expand_dims(flat_state, 0), training=False)
            raw = raw.numpy()[0]
        elif hasattr(policy, '_best_action'):
             
            raw = policy._best_action(flat_state)
        else:
             
            raw = policy(flat_state)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 2:
            raw = raw[0]
        return raw

    def empirical_utility(dp, ap):
        wrapped_dp = lambda w, s: _safe_call(dp, w, s)
        wrapped_ap = lambda w, s: _safe_call(ap, w, s)
        eval_env = MockEnvWrapper(create_env(B, D))
        return -evaluate_interaction(wrapped_dp, wrapped_ap, eval_env, episodes=2)

     
    def _make_uniform_def():
        def _pol(s):
            return np.ones(len(ALERT_TYPES)) / len(ALERT_TYPES)
        return _pol

    def _make_uniform_att():
        def _pol(s):
            return np.ones(len(ATTACK_TYPES)) / len(ATTACK_TYPES)
        return _pol

    container = PolicyContainer(
        [_make_uniform_def()],
        [_make_uniform_att()],
        empirical_utility,
    )

    solver = TimedDoubleOracleSolver(def_oracle, att_oracle, container, wrapper)
    sigma_def, sigma_att = solver.run(
        max_iterations=MAX_DO_ITERATIONS,
        episodes=EPISODES_PER_ORACLE,
    )

    train_time  = solver.elapsed
    n_iters     = solver.iterations_run

     
    def _def_mixed_policy(env_wrapper, flat_state):
        idx = np.random.choice(len(container.def_policies), p=sigma_def)
        pol = container.def_policies[idx]
        return _safe_call(pol, env_wrapper, flat_state)

    def _att_mixed_policy(env_wrapper, flat_state):
        idx = np.random.choice(len(container.att_policies), p=sigma_att)
        pol = container.att_policies[idx]
        return _safe_call(pol, env_wrapper, flat_state)

    eval_env = MockEnvWrapper(create_env(B, D))

    arl_loss     = evaluate_interaction(_def_mixed_policy, _att_mixed_policy, eval_env)
    uniform_loss = evaluate_interaction(defender_uniform,  _att_mixed_policy, eval_env)
    gain_loss    = evaluate_interaction(defender_gain,     _att_mixed_policy, eval_env)

    print(f"  Training time : {train_time:.2f}s | Iterations : {n_iters}")
    print(f"  ARL Loss      : {arl_loss:.4f}")
    print(f"  Uniform Loss  : {uniform_loss:.4f}")
    print(f"  GAIN Loss     : {gain_loss:.4f}")

    return {
        'Oracle'         : oracle_name,
        'B'              : B,
        'D'              : D,
        'Train_Time_s'   : round(train_time, 2),
        'DO_Iterations'  : n_iters,
        'ARL_Loss'       : round(arl_loss, 4),
        'Uniform_Loss'   : round(uniform_loss, 4),
        'GAIN_Loss'      : round(gain_loss, 4),
    }


 
 
 

def run_comparison():
    print("=" * 60)
    print("  DDPG-MIX vs Semi-Gradient n-step Sarsa (SGS) Comparison")
    print(f"  Episodes/oracle={EPISODES_PER_ORACLE}, n-steps={SGS_N_STEPS}, "
          f"bins={SGS_N_BINS}, DO_iters={MAX_DO_ITERATIONS}")
    print("=" * 60)

    results = []

    for scenario in SCENARIOS:
        B, D, label = scenario['B'], scenario['D'], scenario['label']

         
        ddpg_result = run_oracle_experiment(
            oracle_name        = 'DDPG-MIX',
            B=B, D=D,
            oracle_factory_def = lambda B, D: make_ddpg_oracle('defender', B=B, D=D),
            oracle_factory_att = lambda B, D: make_ddpg_oracle('adversary', B=B, D=D),
        )
        results.append(ddpg_result)

         
        sgs_result = run_oracle_experiment(
            oracle_name        = 'SGS (n-step Sarsa)',
            B=B, D=D,
            oracle_factory_def = lambda B, D: make_sgs_oracle('defender', B=B, D=D),
            oracle_factory_att = lambda B, D: make_sgs_oracle('adversary', B=B, D=D),
        )
        results.append(sgs_result)

     
     
     
    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("  FINAL COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("  SUMMARY — Speedup & Loss Delta (DDPG baseline)")
    print("=" * 60)

    for scenario in SCENARIOS:
        B, D = scenario['B'], scenario['D']
        ddpg_row = df[(df['Oracle'] == 'DDPG-MIX') & (df['B'] == B) & (df['D'] == D)].iloc[0]
        sgs_row  = df[(df['Oracle'] == 'SGS (n-step Sarsa)') & (df['B'] == B) & (df['D'] == D)].iloc[0]

        speedup = ddpg_row['Train_Time_s'] / max(sgs_row['Train_Time_s'], 1e-9)
        loss_delta = sgs_row['ARL_Loss'] - ddpg_row['ARL_Loss']

        print(f"\n  Scenario {B=}, {D=}:")
        print(f"    Training speedup (DDPG/SGS) : {speedup:.2f}×")
        print(f"    ARL loss delta  (SGS-DDPG)  : {loss_delta:+.4f}"
              f"  ({'SGS better' if loss_delta < 0 else 'DDPG better or tied'})")

    print("\nFinal Results (creditcard.csv - Paper Parameters)")


if __name__ == "__main__":
    run_comparison()
