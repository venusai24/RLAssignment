"""SGS.py — Semi-Gradient n-step Sarsa Oracle with Linear Function Approximation"""

import numpy as np
import itertools


 
 
 

def _polynomial_features(state: np.ndarray, action: np.ndarray, degree: int = 2) -> np.ndarray:
    """Builds a polynomial basis feature vector phi(s,a)."""
    x = np.concatenate([state, action]).astype(np.float64)
     
    x = np.clip(x, -1e3, 1e3)

    feats = [1.0]   
    feats.extend(x)   

    if degree >= 2:
         
        d = len(x)
        for i in range(d):
            for j in range(i + 1, d):
                feats.append(x[i] * x[j])

    return np.array(feats, dtype=np.float64)


def _feature_dim(input_dim: int, degree: int = 2) -> int:
    """Returns the dimension of the polynomial feature vector."""
    d = input_dim
    linear_terms = d
    cross_terms = d * (d - 1) // 2 if degree >= 2 else 0
    return 1 + linear_terms + cross_terms   


 
 
 

def _build_action_grid(action_dim: int, n_bins: int = 3) -> np.ndarray:
    """Creates a discrete grid of candidate actions."""
    bins = np.linspace(0.0, 1.0, n_bins)
    grid = list(itertools.product(bins, repeat=action_dim))
    return np.array(grid, dtype=np.float64)    


 
 
 

class SGS_Oracle:
    """Semi-Gradient n-step Sarsa Oracle with linear function approximation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        player_role: str,
        n_steps: int = 5,
        alpha: float = 0.01,
        gamma: float = 0.95,
        n_bins: int = 3,
        poly_degree: int = 2,
        action_costs: np.ndarray = None,
        budget: float = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.player_role = player_role
        self.n_steps = n_steps
        self.alpha = alpha
        self.gamma = gamma
        self.n_bins = n_bins
        self.poly_degree = poly_degree
        self.action_costs = action_costs
        self.budget = budget

         
        self.action_grid = _build_action_grid(action_dim, n_bins)
        self.n_candidates = len(self.action_grid)

         
        input_dim = state_dim + action_dim
        self.feat_dim = _feature_dim(input_dim, poly_degree)

         
        self.w = np.random.randn(self.feat_dim) * 0.01

         
        self._step_count = 0
        self._alpha_decay = 1e-4    

     
     
     

    def _phi(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Feature vector φ(s, a)."""
        return _polynomial_features(state, action, self.poly_degree)

    def q_value(self, state: np.ndarray, action: np.ndarray) -> float:
        """Q(s,a) = w · φ(s,a)."""
        return float(np.dot(self.w, self._phi(state, action)))

     
     
     

    def _best_action(self, state: np.ndarray) -> np.ndarray:
        """Returns the greedy action from the discrete candidate set."""
        q_vals = np.array([self.q_value(state, a) for a in self.action_grid])
        best_idx = int(np.argmax(q_vals))
        return self.action_grid[best_idx].copy()

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """ε-greedy action selection over the discrete action grid."""
        if np.random.rand() < epsilon:
            idx = np.random.randint(self.n_candidates)
            return self.action_grid[idx].copy()
        return self._best_action(state)

     
     
     

    def map_continuous_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Projects a raw [0,1]^d action into the feasible budget space.
        Replicates the DDPG projection so the two oracles are comparable.
        """
        if self.player_role == 'defender':
            return raw_action

        elif self.player_role == 'adversary':
            if self.action_costs is None or self.budget is None:
                raise ValueError("Adversary requires action_costs and budget.")
            action_probs = np.array(raw_action, dtype=np.float64)
            expected_cost = np.sum(action_probs * self.action_costs)
            if expected_cost > 0:
                scaling = min(1.0, self.budget / expected_cost)
                return action_probs * scaling
            return action_probs

     
     
     

    def _current_alpha(self) -> float:
        """Decaying step size: α / (1 + t * decay)."""
        return self.alpha / (1.0 + self._step_count * self._alpha_decay)

    def _update_from_trajectory(
        self,
        states: list,
        actions: list,
        rewards: list,
    ):
        """Applies the semi-gradient n-step Sarsa update to the trajectory."""
        T = len(rewards)    
        alpha = self._current_alpha()
        self._step_count += 1

        for tau in range(T):
             
            G = 0.0
            for i in range(self.n_steps):
                t = tau + i
                if t < T:
                    G += (self.gamma ** i) * rewards[t]

             
            bootstrap_t = tau + self.n_steps
            if bootstrap_t < len(states):
                G += (self.gamma ** self.n_steps) * self.q_value(
                    states[bootstrap_t], actions[bootstrap_t]
                )

             
            phi_tau = self._phi(states[tau], actions[tau])
            q_tau = float(np.dot(self.w, phi_tau))
            td_error = G - q_tau
            self.w += alpha * td_error * phi_tau
             
            self.w = np.clip(self.w, -1e6, 1e6)

     
     
     

    def compute_best_response(
        self,
        env,
        opponent_pure_strategies: list,
        opponent_mixed_strategy: np.ndarray,
        episodes: int = 50,
        max_steps: int = 400,
        epsilon: float = 0.1,
    ):
        """Algorithm 1 (paper) implemented with n-step Sarsa."""
        epsilon_schedule = np.linspace(epsilon, 0.02, episodes)   

        for ep in range(episodes):
            eps = epsilon_schedule[ep]
            state = env.reset()

             
            sampled_idx = np.random.choice(
                len(opponent_pure_strategies), p=opponent_mixed_strategy
            )
            pi_minus_v = opponent_pure_strategies[sampled_idx]

            states, actions, rewards = [], [], []

            for step in range(max_steps):
                 
                raw_action = self.select_action(state, epsilon=eps)

                 
                action = self.map_continuous_action(raw_action)

                 
                opponent_action = pi_minus_v(env.get_opponent_state())

                 
                if self.player_role == 'defender':
                    next_state, reward, done, _ = env.step(action, opponent_action)
                else:
                    next_state, def_reward, done, _ = env.step(opponent_action, action)
                    reward = -def_reward    

                states.append(state)
                actions.append(raw_action)
                rewards.append(reward)

                state = next_state
                if done:
                    break

             
            self._update_from_trajectory(states, actions, rewards)

         
        return self

     
     
     

    def __call__(self, state):
        """
        Allows SGS_Oracle to be used as a policy callable.
        Returns greedy action as a flat 1-D array.
        """
        if hasattr(state, 'numpy'):
            state = state.numpy()
        state = np.asarray(state).flatten()
        return self._best_action(state)
