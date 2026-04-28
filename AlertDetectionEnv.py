import math
import random

class AlertDetectionEnv:
    def __init__(self, alert_types, attack_types, def_budget, att_budget, 
                 investigation_costs, attack_costs, attack_losses, alert_probs, false_alert_probs):
        """
        Initializes the Attack Detection Environment.
        
        :param alert_types: List of alert types (T)
        :param attack_types: List of attack types (A)
        :param def_budget: Defender's budget (B)
        :param att_budget: Adversary's budget (D)
        :param investigation_costs: Dict mapping t in T to cost C_t
        :param attack_costs: Dict mapping a in A to cost E_a
        :param attack_losses: Dict mapping a in A to loss L_a
        :param alert_probs: Dict mapping (a, t) to a function that returns number of alerts generated
        :param false_alert_probs: Dict mapping t to a function that returns number of false alerts generated
        """
        self.T = alert_types
        self.A = attack_types
        self.B = def_budget
        self.D = att_budget
        self.C = investigation_costs
        self.E = attack_costs
        self.L = attack_losses
        self.P = alert_probs
        self.F = false_alert_probs
        
        self.reset()

    def reset(self):
        """Resets the environment to the initial state <N(0), M(0), S(0)> = <0, 0, 0>"""
        self.N = {t: 0 for t in self.T}
        self.M = {a: 0 for a in self.A}
        self.S = {a: {t: 0 for t in self.T} for a in self.A}
        return self.get_state()

    def get_state(self):
        """Returns the current state tuple"""
        return self.N.copy(), self.M.copy(), self.S.copy()

    def _check_constraints(self, alpha_plus, alpha_minus):
        """Validates actions against attacker and defender constraints."""
         
        def_cost = sum(self.C[t] * alpha_plus.get(t, 0) for t in self.T)
        if def_cost > self.B + 1e-9:
            raise ValueError(f"Defender budget exceeded: {def_cost} > {self.B}")
            
         
        for t in self.T:
            if alpha_plus.get(t, 0) > self.N[t]:
                raise ValueError(f"Cannot investigate more {t} alerts than exist.")

         
        att_cost = sum(self.E[a] * alpha_minus.get(a, 0) for a in self.A)
        if att_cost > self.D + 1e-9:
            raise ValueError(f"Attacker budget exceeded: {att_cost} > {self.D}")

    def step(self, alpha_plus, alpha_minus):
        """
        Executes one time period (k).
        :param alpha_plus: Dict mapping t to number of alerts to investigate (Defender Action)
        :param alpha_minus: Dict mapping a to binary indicator of attack execution (Attacker Action)
        """
         
        alpha_plus = {t: int(v) for t, v in alpha_plus.items()}
        
        self._check_constraints(alpha_plus, alpha_minus)

         
        M_tilde = {a: 0 for a in self.A}
        
        for a in self.A:
            if self.M[a] == 1:
                 
                p_a = 1.0
                for t in self.T:
                    n_t = self.N[t]
                    r_t = alpha_plus.get(t, 0)
                    s_at = self.S[a][t]
                    
                     
                    pool_remaining = max(0, n_t - s_at)
                    
                    if r_t > pool_remaining:
                         
                         
                        p_a = 0.0
                        break
                    
                     
                    numerator = math.comb(pool_remaining, r_t)
                    denominator = math.comb(n_t, r_t)
                    
                    if denominator > 0:
                        p_a *= (numerator / denominator)
                    else:
                        p_a *= 1.0
                        
                 
                if random.random() < p_a:
                    M_tilde[a] = 1

         
        reward = -sum(self.L[a] * M_tilde[a] for a in self.A)

         
        for t in self.T:
            self.N[t] -= alpha_plus.get(t, 0)

         
         
        for a in self.A:
            prob = alpha_minus.get(a, 0)
            self.M[a] = 1 if random.random() < prob else 0
             
            for t in self.T:
                self.S[a][t] = 0

         
         
        for a in self.A:
            if self.M[a] == 1:
                for t in self.T:
                     
                    generated_alerts = self.P[(a, t)]() 
                    self.S[a][t] = generated_alerts
                    self.N[t] += generated_alerts
        
         
        for t in self.T:
             
            self.N[t] += self.F[t]()

        return self.get_state(), reward