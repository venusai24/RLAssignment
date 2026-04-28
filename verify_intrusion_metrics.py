import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from AlertDetectionEnv import AlertDetectionEnv
from DDPG import DDPG_MIX_Oracle
from DoubleOracleSolver import DoubleOracleSolver, PolicyContainer


ALERT_TYPES = [
    'attempted-recon',         
    'attempted-user',          
    'bad-unknown',             
    'misc-activity',           
    'not-suspicious',          
    'policy-violation',        
    'protocol-command-decode', 
]

 
ATTACK_TYPES = [
    'Brute Force',    
    'Botnet',         
    'DoS',            
    'Heartbleed',     
    'Infiltration',   
    'PortScan',       
    'Web Attack',     
]

 
 
 
 
 
 
ATTACK_ALERT_COUNTS = {
     
    'Brute Force': [1230,  0,   0,    0,    0,    0,   0],
    'Botnet':      [   0,  4,   2,  106,    0,   54,   0],
    'DoS':         [   0,  0,   0,    0,    0,   24,   0],
    'Heartbleed':  [   0,  0,   4,    0,   10,    0,   0],
    'Infiltration':[  710,  2, 862,   12,    0,   80, 600],
    'PortScan':    [  138,  0, 320,   30,    0,    0,   0],
    'Web Attack':  [   0,  0,   6,    0,    0,    0,   0],
}

 
FALSE_ALERT_MEANS = {
    'attempted-recon':         700,
    'attempted-user':         4400,
    'bad-unknown':             100,
    'misc-activity':           700,
    'not-suspicious':         1700,
    'policy-violation':        400,
    'protocol-command-decode':1000,
}

 
ATTACKER_COSTS = {
    'Brute Force':  120,
    'Botnet':        60,
    'DoS':           74,
    'Heartbleed':    20,
    'Infiltration':  52,
    'PortScan':      80,
    'Web Attack':    62,
}

 
 
DEFENDER_LOSSES = {
    'Brute Force':  3.9,    
    'Botnet':       6.5,    
    'DoS':          4.0,    
    'Heartbleed':   3.7,    
    'Infiltration': 1.8,    
    'PortScan':     1.3,    
    'Web Attack':   2.6,    
}

 
INVESTIGATION_COSTS = {t: 1.0 for t in ALERT_TYPES}


 
def _build_p_matrix():
 
    P = {}
    for a, counts in ATTACK_ALERT_COUNTS.items():
        for j, t in enumerate(ALERT_TYPES):
            count = counts[j]
             
            P[(a, t)] = (lambda c=count: c)
    return P


def _build_f_functions():
    F = {}
    for t in ALERT_TYPES:
        mean = FALSE_ALERT_MEANS[t]
        F[t] = (lambda m=mean: int(np.random.poisson(m)))
    return F


 
def create_env(B: float, D: float) -> AlertDetectionEnv:
    return AlertDetectionEnv(
        alert_types=ALERT_TYPES,
        attack_types=ATTACK_TYPES,
        def_budget=B,
        att_budget=D,
        investigation_costs=INVESTIGATION_COSTS,
        attack_costs=ATTACKER_COSTS,
        attack_losses=DEFENDER_LOSSES,
        alert_probs=_build_p_matrix(),
        false_alert_probs=_build_f_functions(),
    )


 
class MockEnvWrapper:
    def __init__(self, env: AlertDetectionEnv):
        self.env = env
        self.state: tuple = None

    def reset(self) -> np.ndarray:
        self.state = self.env.reset()
        return self._flatten(self.state)

    def _flatten(self, state_tuple) -> np.ndarray:
        N, M, S = state_tuple
        return np.concatenate([
            [N[t] for t in self.env.T],
            [M[a] for a in self.env.A],
            [S[a][t] for a in self.env.A for t in self.env.T],
        ]).astype(np.float32)

    def step(self, def_action: np.ndarray, att_action: np.ndarray):
        """
        def_action : fractions of budget B allocated to each alert type
                     (output of defender policy, length |T|)
        att_action : probability of executing each attack
                     (output of adversary policy, length |A|)
        """
        alpha_plus  = {t: def_action[i] * self.env.B
                       for i, t in enumerate(self.env.T)}
        alpha_minus = {a: float(att_action[i])
                       for i, a in enumerate(self.env.A)}

        next_state_tuple, reward = self.env.step(alpha_plus, alpha_minus)
        self.state = next_state_tuple
        return self._flatten(next_state_tuple), reward, False, {}

    def get_opponent_state(self) -> np.ndarray:
        return self._flatten(self.state)


 
def evaluate_interaction(def_policy, att_policy, env_wrapper: MockEnvWrapper,
                         episodes: int = 10) -> float:
    """
    Rolls out def_policy vs att_policy for `episodes` episodes of 50 steps
    each and returns the mean *expected loss* of the defender
    (i.e., −mean discounted reward, same metric used in Figures 4–7).

    Policy signatures:
        def_policy(wrapper, flat_state) → np.ndarray of length |T|
        att_policy(wrapper, flat_state) → np.ndarray of length |A|
    """
    gamma = 0.95    
    all_rewards = []

    for _ in range(episodes):
        flat_state = env_wrapper.reset()
        episode_reward = 0.0

        for k in range(50):
            d_act = def_policy(env_wrapper, flat_state)
            if hasattr(d_act, 'numpy'):
                d_act = d_act.numpy()
            if d_act.ndim == 2:
                d_act = d_act[0]

            a_act = att_policy(env_wrapper, flat_state)
            if hasattr(a_act, 'numpy'):
                a_act = a_act.numpy()
            if a_act.ndim == 2:
                a_act = a_act[0]

            flat_state, reward, _, _ = env_wrapper.step(d_act, a_act)
            episode_reward += (gamma ** k) * reward

        all_rewards.append(episode_reward)

     
    return -np.mean(all_rewards)


 
def train_arl_solver(B: float, D: float,
                     max_iterations: int = 5) -> tuple:
    """
    Runs the double-oracle / DDPG-MIX framework for the intrusion-detection
    case study and returns (sigma_def, sigma_att, container).

    Network architecture per Table II for *intrusion detection*:
        Policy network: hidden=32, Tanh, Xavier
        Value  network: hidden=64, ReLU, HeNormal

    State dim = 7 (N) + 7 (M) + 7×7 (S) = 63
    """
    env = create_env(B, D)
    wrapper = MockEnvWrapper(env)

    n_alerts  = len(ALERT_TYPES)    
    n_attacks = len(ATTACK_TYPES)   
    state_dim = n_alerts + n_attacks + n_attacks * n_alerts   

    att_costs_array = np.array([ATTACKER_COSTS[a] for a in ATTACK_TYPES],
                                dtype=np.float32)

     
    def empirical_utility(dp, ap):
        wrapped_dp = lambda w, s: dp(s)
        wrapped_ap = lambda w, s: ap(s)
        return -evaluate_interaction(
            wrapped_dp, wrapped_ap,
            MockEnvWrapper(create_env(B, D)),
            episodes=2,
        )

    def_oracle = DDPG_MIX_Oracle(
        state_dim=state_dim,
        action_dim=n_alerts,
        player_role='defender',
        domain='intrusion',    
        budget=B,
    )
    att_oracle = DDPG_MIX_Oracle(
        state_dim=state_dim,
        action_dim=n_attacks,
        player_role='adversary',
        domain='intrusion',
        action_costs=att_costs_array,
        budget=D,
    )

     
    init_def = [lambda s: np.ones(n_alerts,  dtype=np.float32) / n_alerts]
    init_att = [lambda s: np.ones(n_attacks, dtype=np.float32) / n_attacks]

    container = PolicyContainer(init_def, init_att, empirical_utility)
    solver    = DoubleOracleSolver(def_oracle, att_oracle, container, wrapper)

    sigma_def, sigma_att = solver.run(max_iterations=max_iterations)
    return sigma_def, sigma_att, container


 

def worker_fig4(task: dict) -> tuple:
    """
    Figure 4: Defender loss when it knows the attack budget.
        Left panel  – vary B ∈ {500, 1000, 1500}, fixed D = 120
        Right panel – vary D ∈ {60, 120, 180},    fixed B = 1000
    Only ARL is evaluated.
    """
    B          = task['B']
    D          = task['D']
    panel      = task['panel']

    sigma_def, sigma_att, container = train_arl_solver(B, D)

    n_def = len(container.def_policies)
    n_att = len(container.att_policies)

    def_mixed = lambda w, s: container.def_policies[
        np.random.choice(n_def, p=sigma_def)](s)
    att_mixed = lambda w, s: container.att_policies[
        np.random.choice(n_att, p=sigma_att)](s)

    eval_wrapper = MockEnvWrapper(create_env(B, D))
    loss = evaluate_interaction(def_mixed, att_mixed, eval_wrapper, episodes=10)

    return ('fig4', {'panel': panel, 'B': B, 'D': D, 'ARL': loss})


def worker_fig5(task: dict) -> tuple:
    """
    Figure 5: Defender loss when *uncertain* about attack budget.
        Defender always estimates D = 120 (trains with D_est = 120).
        Actual D ∈ {60, 120, 180}.
        Left panel  – B = 500
        Right panel – B = 1500
    Only ARL is evaluated.
    """
    B        = task['B']
    D_actual = task['D_actual']
    D_est    = 120    

     
    sigma_def_est, _, cont_est = train_arl_solver(B, D_est)
    n_def = len(cont_est.def_policies)
    def_mixed = lambda w, s: cont_est.def_policies[
        np.random.choice(n_def, p=sigma_def_est)](s)

     
    _, sigma_att_act, cont_act = train_arl_solver(B, D_actual)
    n_att = len(cont_act.att_policies)
    att_mixed = lambda w, s: cont_act.att_policies[
        np.random.choice(n_att, p=sigma_att_act)](s)

    eval_wrapper = MockEnvWrapper(create_env(B, D_actual))
    loss = evaluate_interaction(def_mixed, att_mixed, eval_wrapper, episodes=10)

    return ('fig5', {'B': B, 'D_est': D_est, 'D_actual': D_actual, 'ARL': loss})


def worker_fig6(task: dict) -> tuple:
    """
    Figure 6: ARL loss matrix when defender has *different* estimates of
              the attack budget.
        Defender estimates ∈ {60, 120, 180}
        Actual attacker budget ∈ {60, 120, 180}
        Fixed B = 1000 (mid-range, paper uses same B for Fig 6 matrix).
    """
    B        = 1000
    D_est    = task['D_est']
    D_actual = task['D_actual']

    sigma_def_est, _, cont_est = train_arl_solver(B, D_est)
    n_def = len(cont_est.def_policies)
    def_mixed = lambda w, s: cont_est.def_policies[
        np.random.choice(n_def, p=sigma_def_est)](s)

    _, sigma_att_act, cont_act = train_arl_solver(B, D_actual)
    n_att = len(cont_act.att_policies)
    att_mixed = lambda w, s: cont_act.att_policies[
        np.random.choice(n_att, p=sigma_att_act)](s)

    eval_wrapper = MockEnvWrapper(create_env(B, D_actual))
    loss = evaluate_interaction(def_mixed, att_mixed, eval_wrapper, episodes=10)

    return ('fig6', {'D_est': D_est, 'D_actual': D_actual, 'ARL': loss})


def worker_fig7(task: dict) -> tuple:
    """
    Figure 7: ARL defender vs different *attacker policies* when the
              defender knows the budget but is uncertain of attacker
              rationality.
        Attack budget D = 120 (fixed).
        Left panel  – B = 500
        Right panel – B = 1500
    Only the ARL defender is evaluated (ARL row of the figure).
    The attacker is also trained with ARL (i.e., full adversarial RL row).
    """
    B = task['B']
    D = 120   

    sigma_def, sigma_att, container = train_arl_solver(B, D)

    n_def = len(container.def_policies)
    n_att = len(container.att_policies)

    def_mixed = lambda w, s: container.def_policies[
        np.random.choice(n_def, p=sigma_def)](s)
    att_mixed = lambda w, s: container.att_policies[
        np.random.choice(n_att, p=sigma_att)](s)

    eval_wrapper = MockEnvWrapper(create_env(B, D))
    loss = evaluate_interaction(def_mixed, att_mixed, eval_wrapper, episodes=10)

    return ('fig7', {'B': B, 'D': D, 'ARL': loss})


 

def run_full_verification():
    """
    Runs all workers in parallel and prints results that correspond to
    Figures 4–7 of Case Study I (Intrusion Detection) in the paper.

    Only the ARL method is verified; baseline methods (Uniform, Suricata)
    are deliberately omitted.
    """
    tasks: list[tuple] = []

     
     
    for B in [500, 1000, 1500]:
        tasks.append((worker_fig4, {'panel': 'left',  'B': B, 'D': 120}))
     
    for D in [60, 120, 180]:
        tasks.append((worker_fig4, {'panel': 'right', 'B': 1000, 'D': D}))

     
    for B in [500, 1500]:
        for D_actual in [60, 120, 180]:
            tasks.append((worker_fig5, {'B': B, 'D_actual': D_actual}))

     
    for D_est in [60, 120, 180]:
        for D_actual in [60, 120, 180]:
            tasks.append((worker_fig6, {'D_est': D_est, 'D_actual': D_actual}))

     
    for B in [500, 1500]:
        tasks.append((worker_fig7, {'B': B}))

     
    fig4_results  = []
    fig5_results  = []
    fig6_matrix   = np.zeros((3, 3))
    D_idx         = {60: 0, 120: 1, 180: 2}
    fig7_results  = []

    print("=" * 60)
    print(" Case Study I: Intrusion Detection – ARL Verification")
    print("=" * 60)
    print(f"Submitting {len(tasks)} tasks to ProcessPoolExecutor …\n")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, arg) for func, arg in tasks]
        for future in futures:
            tag, res = future.result()

            if tag == 'fig4':
                fig4_results.append(res)
                print(f"  [Fig 4] panel={res['panel']}  B={res['B']:5d}  "
                      f"D={res['D']:4d}  ARL_loss={res['ARL']:.2f}")

            elif tag == 'fig5':
                fig5_results.append(res)
                print(f"  [Fig 5] B={res['B']:5d}  D_est={res['D_est']}  "
                      f"D_actual={res['D_actual']}  ARL_loss={res['ARL']:.2f}")

            elif tag == 'fig6':
                r, c = D_idx[res['D_actual']], D_idx[res['D_est']]
                fig6_matrix[r, c] = res['ARL']
                print(f"  [Fig 6] D_est={res['D_est']}  "
                      f"D_actual={res['D_actual']}  ARL_loss={res['ARL']:.2f}")

            elif tag == 'fig7':
                fig7_results.append(res)
                print(f"  [Fig 7] B={res['B']:5d}  D={res['D']}  "
                      f"ARL_loss={res['ARL']:.2f}")

     
    print("\n" + "=" * 60)
    print("FIGURE 4 – ARL Loss (defender knows attack budget)")
    print("=" * 60)
    df4 = pd.DataFrame(fig4_results)
     
    print("\n[Left panel] Fixed D = 120, varying B:")
    left = df4[df4['panel'] == 'left'].sort_values('B')[['B', 'D', 'ARL']]
    print(left.to_string(index=False))
     
    print("\n[Right panel] Fixed B = 1000, varying D:")
    right = df4[df4['panel'] == 'right'].sort_values('D')[['B', 'D', 'ARL']]
    print(right.to_string(index=False))

    print("\n" + "=" * 60)
    print("FIGURE 5 – ARL Loss (uncertain attack budget, D_est = 120)")
    print("=" * 60)
    df5 = pd.DataFrame(fig5_results)
    for B in [500, 1500]:
        panel = df5[df5['B'] == B].sort_values('D_actual')[
            ['B', 'D_est', 'D_actual', 'ARL']]
        print(f"\n[B = {B}]")
        print(panel.to_string(index=False))

    print("\n" + "=" * 60)
    print("FIGURE 6 – ARL Loss Matrix  (B = 1000)")
    print("         Columns: Estimated D → [60, 120, 180]")
    print("         Rows:    Actual D    → [60, 120, 180]")
    print("=" * 60)
    df6 = pd.DataFrame(
        fig6_matrix,
        index  =['D_actual=60', 'D_actual=120', 'D_actual=180'],
        columns=['D_est=60',    'D_est=120',    'D_est=180'],
    )
    print(df6.round(2).to_string())

    print("\n" + "=" * 60)
    print("FIGURE 7 – ARL Defender vs ARL Attacker (D = 120)")
    print("=" * 60)
    df7 = pd.DataFrame(fig7_results).sort_values('B')[['B', 'D', 'ARL']]
    print(df7.to_string(index=False))
    print()


if __name__ == "__main__":
    run_full_verification()
