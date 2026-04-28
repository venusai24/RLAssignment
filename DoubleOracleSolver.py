import numpy as np
from scipy.optimize import linprog

class PolicyContainer:
    """Stores the policies for the defender and attacker, and their utility matrix"""
    def __init__(self, init_def_policies, init_att_policies, utility_evaluator):
        self.def_policies = init_def_policies
        self.att_policies = init_att_policies
        self.utility_evaluator = utility_evaluator
        self.utility_matrix = None
        self._initialize_matrix()

    def _initialize_matrix(self):
        """Computes initial U matrix where U[i,j] is the defender's utility."""
        n_def = len(self.def_policies)
        n_att = len(self.att_policies)
        self.utility_matrix = np.zeros((n_def, n_att))
        
        for i in range(n_def):
            for j in range(n_att):
                self.utility_matrix[i, j] = self.utility_evaluator(
                    self.def_policies[i], self.att_policies[j]
                )

    def add_policies_and_update_matrix(self, new_def_policy, new_att_policy):
        """Adds new best response policies to Pi_{+1} and Pi_{-1} and updates U"""
        if new_def_policy is not None:
            self.def_policies.append(new_def_policy)
        if new_att_policy is not None:
            self.att_policies.append(new_att_policy)
            
         
        self._initialize_matrix()

class DoubleOracleSolver:
    """Orchestrates the Double Oracle algorithm to find an approximate MSNE"""
    def __init__(self, def_oracle, att_oracle, policy_container, env=None):
        self.def_oracle = def_oracle
        self.att_oracle = att_oracle
        self.container = policy_container
        self.env = env

    def solve_msne(self, utility_matrix):
        """Solves the zero-sum game matrix to find the MSNE."""
        num_strategies, num_opponent_strategies = utility_matrix.shape
        
         
        c = np.zeros(num_strategies + 1)
        c[-1] = -1.0 
        
         
         
        A_ub = np.zeros((num_opponent_strategies, num_strategies + 1))
        A_ub[:, :-1] = -utility_matrix.T
        A_ub[:, -1] = 1.0
        
        b_ub = np.zeros(num_opponent_strategies)
        
         
        A_eq = np.zeros((1, num_strategies + 1))
        A_eq[0, :-1] = 1.0
        b_eq = np.array([1.0])
        
         
        bounds = [(0, None) for _ in range(num_strategies)] + [(None, None)]
        
         
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            raise ValueError("Linear program failed to converge.")
            
        sigma_v = res.x[:-1]
        u_v_star = res.x[-1]
        
        return sigma_v, u_v_star

    def evaluate_best_response_utility(self, new_policy, mixed_strategy, opponent_policies, is_defender):
        """Calculates expected utility of a pure strategy against an opponent's mixed strategy"""
        expected_utility = 0.0
        for i, opp_policy in enumerate(opponent_policies):
            prob = mixed_strategy[i]
            if prob > 0:
                if is_defender:
                    u = self.container.utility_evaluator(new_policy, opp_policy)
                else:
                     
                    u = -self.container.utility_evaluator(opp_policy, new_policy)
                expected_utility += prob * u
        return expected_utility

    def run(self, max_iterations=50):
        """Executes the iterative Double Oracle loop"""
        iteration = 0
        
        while iteration < max_iterations:
             
             
            sigma_def, u_def_eq = self.solve_msne(self.container.utility_matrix)
             
            sigma_att, u_att_eq = self.solve_msne(-self.container.utility_matrix.T)
            
             
            new_pi_def = self.def_oracle.compute_best_response(self.env, self.container.att_policies, sigma_att)
            new_pi_att = self.att_oracle.compute_best_response(self.env, self.container.def_policies, sigma_def)
            
             
            u_new_def = self.evaluate_best_response_utility(
                new_pi_def, sigma_att, self.container.att_policies, is_defender=True
            )
            u_new_att = self.evaluate_best_response_utility(
                new_pi_att, sigma_def, self.container.def_policies, is_defender=False
            )
            
             
            def_improves = u_new_def > u_def_eq + 1e-5
            att_improves = u_new_att > u_att_eq + 1e-5
            
            if not def_improves and not att_improves:
                print(f"Converged after {iteration} iterations.")
                break
                
             
            add_def = new_pi_def if def_improves else None
            add_att = new_pi_att if att_improves else None
            
            self.container.add_policies_and_update_matrix(add_def, add_att)
            iteration += 1
            
        return sigma_def, sigma_att