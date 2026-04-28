import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, HeNormal

class DDPG_MIX_Oracle:
    def __init__(self, state_dim, action_dim, player_role, domain='intrusion', 
                 action_costs=None, budget=None, gamma=0.95, tau=0.005, buffer_capacity=40000):
        """
        Initializes the DDPG-MIX Oracle.
        player_role: 'defender' or 'adversary'
        domain: 'fraud' or 'intrusion' (determines hidden units)
        action_costs: Array of costs for each action (needed for adversary projection)
        budget: Total budget constraint (D for adversary, B for defender)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.player_role = player_role
        self.domain = domain
        self.action_costs = action_costs
        self.budget = budget
        
        self.gamma = gamma
        self.tau = tau
        
         
        if self.domain == 'fraud':
            self.actor_hidden = 16
            self.critic_hidden = 32
        else:  
            self.actor_hidden = 32
            self.critic_hidden = 64
            
         
        self.actor = self._build_actor()
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        
        self.critic = self._build_critic()
        self.target_critic = self._build_critic()
        self.target_critic.set_weights(self.critic.get_weights())
        
         
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        
         
        self.buffer = ReplayBuffer(buffer_capacity)

    def _build_actor(self):
        """Builds Policy Network per Table II """
        inputs = Input(shape=(self.state_dim,))
        
         
        x = Dense(self.actor_hidden, activation='tanh', 
                  kernel_initializer=GlorotUniform())(inputs)
        
         
        outputs = Dense(self.action_dim, activation='sigmoid', 
                        kernel_initializer=GlorotUniform())(x)
        
        return Model(inputs, outputs)

    def _build_critic(self):
        """Builds Value Network per Table II """
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        
         
        concat = Concatenate()([state_input, action_input])
        
         
        x = Dense(self.critic_hidden, activation='relu', 
                  kernel_initializer=HeNormal())(concat)
        
         # Output layer: Linear activation (None) to allow negative Q-values
        # Note: Paper Table II mentions ReLU, but given negative rewards, 
        # a linear output is required for the network to converge.
        outputs = Dense(1, activation=None, kernel_initializer=HeNormal())(x)
        
        return Model([state_input, action_input], outputs)

    def map_continuous_action(self, raw_action):
        """
        Implements the continuous action mapping .
        raw_action: the sigmoid output from the actor network (range [0,1]).
        """
        if self.player_role == 'defender':
             
             
            return raw_action 
            
        elif self.player_role == 'adversary':
             
            if self.action_costs is None or self.budget is None:
                raise ValueError("Adversary requires action_costs and budget to be set.")
            
            action_probs = np.array(raw_action)
            expected_cost = np.sum(action_probs * self.action_costs)
            
            if expected_cost > 0:
                 
                scaling_factor = min(1.0, self.budget / expected_cost)
                projected_action = action_probs * scaling_factor
            else:
                projected_action = action_probs
                
            return projected_action

    @tf.function
    def update_networks(self, state_batch, action_batch, reward_batch, next_state_batch):
        """Updates Actor and Critic using a sampled minibatch [cite: 904-910]"""
         
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            target_q = self.target_critic([next_state_batch, target_actions], training=True)
            
             
            y = reward_batch + self.gamma * target_q
            
            current_q = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - current_q))
            
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
         
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
             
            actor_loss = -tf.reduce_mean(self.critic([state_batch, actions], training=True))
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
         
        for target_weights, weights in zip(self.target_actor.variables, self.actor.variables):
            target_weights.assign(self.tau * weights + (1.0 - self.tau) * target_weights)
            
        for target_weights, weights in zip(self.target_critic.variables, self.critic.variables):
            target_weights.assign(self.tau * weights + (1.0 - self.tau) * target_weights)

    def compute_best_response(self, env, opponent_pure_strategies, opponent_mixed_strategy, 
                              episodes=500, max_steps=400, epsilon=0.1):
        """
        Algorithm 1: Compute pure-strategy best response to a mixed strategy .
        opponent_mixed_strategy (\sigma_{-v}) is a list of probabilities.
        opponent_pure_strategies (\Pi_{-v}) is a list of policy objects/functions.
        """
        for episode in range(episodes):  
            state = env.reset()  
            
             
            sampled_idx = np.random.choice(len(opponent_pure_strategies), p=opponent_mixed_strategy)
            pi_minus_v = opponent_pure_strategies[sampled_idx]
            
            for step in range(max_steps):  
                 
                if np.random.rand() < epsilon:
                    raw_action = np.random.uniform(0, 1, size=(self.action_dim,))
                else:
                    raw_action = self.actor(np.expand_dims(state, axis=0))[0].numpy()
                
                 
                action = self.map_continuous_action(raw_action)
                
                 
                opponent_action = pi_minus_v(env.get_opponent_state()) 
                
                 
                if self.player_role == 'defender':
                    next_state, reward, done, _ = env.step(action, opponent_action)
                else:
                     
                    next_state, def_reward, done, _ = env.step(opponent_action, action)
                    reward = -def_reward
                
                 
                self.buffer.record((state, raw_action, reward, next_state))
                
                 
                if self.buffer.get_length() >= 64:  
                    state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample_batch(64)
                    self.update_networks(state_batch, action_batch, reward_batch, next_state_batch)
                
                state = next_state
                if done:
                    break
                    
        return self.actor  

class ReplayBuffer:
    """Simple Replay Buffer to store experiences D [cite: 869]"""
    def __init__(self, capacity=40000):
        self.capacity = capacity
        self.buffer = []
        
    def record(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)
        
    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        state_batch = tf.convert_to_tensor([b[0] for b in batch], dtype=tf.float32)
        action_batch = tf.convert_to_tensor([b[1] for b in batch], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor([[b[2]] for b in batch], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor([b[3] for b in batch], dtype=tf.float32)
        
        return state_batch, action_batch, reward_batch, next_state_batch
        
    def get_length(self):
        return len(self.buffer)