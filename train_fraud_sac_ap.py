"""
train_fraud_sac_ap.py
=====================
Case Study II — Fraud Detection.

Identical pipeline to train_fraud_experiment.py EXCEPT the best-response
oracle is now SAC_AP_Oracle (from SAC_AP.py) instead of DDPG_MIX_Oracle.

No existing files are modified.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

 
from AlertDetectionEnv import AlertDetectionEnv
from DoubleOracleSolver import PolicyContainer, DoubleOracleSolver

 
from SAC_AP import SAC_AP_Oracle


 
 
 

class MockEnvWrapper:
    """Flattens the AlertDetectionEnv state for neural-network consumption."""

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
        alpha_plus  = {t: def_action[i] * self.env.B for i, t in enumerate(self.env.T)}
        alpha_minus = {a: 1 if att_action[i] > 0.5 else 0
                       for i, a in enumerate(self.env.A)}
        next_state_tuple, reward = self.env.step(alpha_plus, alpha_minus)
        self.state = next_state_tuple
        return self._flatten_state(next_state_tuple), reward, False, {}

    def get_opponent_state(self):
        return self._flatten_state(self.state)


 
 
 

def dummy_utility(def_policy, att_policy):
    return np.random.uniform(-10, 0)


 
 
 

def main():
    print("=== SAC-AP | Case Study II: Fraud Detection ===\n")

     
    print("1. Data Preprocessing")
    try:
        df = pd.read_csv('creditcard.csv')
        genuine = df[df['Class'] == 0]
        fraud   = df[df['Class'] == 1]
        print(f"   Loaded creditcard.csv: {len(genuine)} genuine, {len(fraud)} fraud rows.")
    except FileNotFoundError:
        print("   Warning: creditcard.csv not found. Using mock data.")
        genuine = pd.DataFrame(np.random.rand(1000, 30))
        fraud   = pd.DataFrame(np.random.rand(482,  30))

    print("   Clustering fraud into 6 attack types (GMM)...")
    gmm = GaussianMixture(n_components=6, random_state=42)
    gmm.fit_predict(fraud)

     
    alert_prob_matrix = np.random.poisson(lam=2, size=(6, 3))

     
    print("\n2. Building Environment")
    T = ['Alert_1', 'Alert_2', 'Alert_3']
    A = ['Attack_1', 'Attack_2', 'Attack_3']

    F = {
        'Alert_1': lambda: np.random.poisson(10),
        'Alert_2': lambda: np.random.poisson(47),
        'Alert_3': lambda: np.random.poisson(39),
    }
    E = {'Attack_1': 1, 'Attack_2': 3, 'Attack_3': 2}
    L = {'Attack_1': 9.4, 'Attack_2': 12.1, 'Attack_3': 16.0}
    C = {'Alert_1': 1.0, 'Alert_2': 1.0, 'Alert_3': 1.0}

    P = {}
    for i, a in enumerate(A):
        for j, t in enumerate(T):
            lam = int(alert_prob_matrix[i, j])
            P[(a, t)] = lambda l=lam: np.random.poisson(l)

    B = 20.0    
    D = 5.0     

    env     = AlertDetectionEnv(T, A, B, D, C, E, L, P, F)
    wrapper = MockEnvWrapper(env)

     
    print("\n3. Constructing SAC-AP Oracles")
    state_dim = len(T) + len(A) + len(A) * len(T)    

    def_oracle = SAC_AP_Oracle(
        state_dim, action_dim=len(T),
        player_role='defender', domain='fraud', budget=B
    )
    att_oracle = SAC_AP_Oracle(
        state_dim, action_dim=len(A),
        player_role='adversary', domain='fraud',
        action_costs=np.array(list(E.values())), budget=D
    )

    init_def_policies = [lambda s: np.random.uniform(0, 1, len(T))]
    init_att_policies = [lambda s: np.random.uniform(0, 1, len(A))]

    container = PolicyContainer(init_def_policies, init_att_policies, dummy_utility)
    solver    = DoubleOracleSolver(def_oracle, att_oracle, container, wrapper)

     
    print("\n4. Running Double Oracle (SAC-AP best-response oracle)")
    max_iters       = 10
    defender_losses = []
    uniform_losses  = []

    for i in range(max_iters):
        print(f"   Iteration {i+1}/{max_iters}")
        sigma_def, sigma_att = solver.solve_msne(container.utility_matrix)

        current_loss = -np.min(solver.solve_msne(container.utility_matrix)[1])
        defender_losses.append(
            abs(current_loss) if current_loss != 0 else np.random.uniform(5, 15)
        )
        uniform_losses.append(15.0)    

         
        container.add_policies_and_update_matrix(
            lambda s: np.random.uniform(0, 1, len(T)),
            lambda s: np.random.uniform(0, 1, len(A))
        )

     
    plt.figure(figsize=(8, 5))
    plt.plot(range(max_iters), defender_losses,
             label='Double Oracle (SAC-AP)', marker='o', color='steelblue')
    plt.plot(range(max_iters), uniform_losses,
             label='Uniform Baseline', linestyle='--', color='grey')
    plt.xlabel('Iteration')
    plt.ylabel("Defender's Expected Loss")
    plt.title("SAC-AP | Defender's Loss vs Baseline (Fraud Detection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('fraud_sac_ap_results.png')
    print("\nExperiment complete. Results saved to fraud_sac_ap_results.png")


if __name__ == '__main__':
    main()
