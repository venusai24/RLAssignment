import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, HeNormal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
LOG_ALPHA_MIN = -5.0
LOG_ALPHA_MAX = 2.0
Q_CLIP_MIN = -500.0
Q_CLIP_MAX = 0.0
GRAD_CLIP_NORM = 1.0
EPS = 1e-6


class SAC_AP_Oracle:
    """
    Soft Actor-Critic Best Response Oracle (SAC-AP).

    Replaces DDPG_MIX_Oracle in the Double Oracle loop.
    Key differences from DDPG:
      - Double Q-networks (Q_theta1, Q_theta2) for conservative Q-estimate
      - Entropy-regularised objective: J(pi) = E[r + gamma*(min_Q - alpha*log_pi)]
      - Automatic temperature tuning: alpha adjusted to maintain target entropy
      - Reparameterisation trick with tanh squashing → actions in [0, 1]
      - NaN prevention: log_std clamping, gradient clipping, Q-target clipping,
        safe log operations, log_alpha clamping
    """

    def __init__(self, state_dim, action_dim, player_role, domain='fraud',
                 action_costs=None, budget=None, gamma=0.95, tau=0.005,
                 buffer_capacity=40000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.player_role = player_role
        self.domain = domain
        self.action_costs = action_costs
        self.budget = budget
        self.gamma = gamma
        self.tau = tau

        if domain == 'fraud':
            self.actor_hidden = 16
            self.critic_hidden = 32
        else:
            self.actor_hidden = 32
            self.critic_hidden = 64

         
        self.actor = self._build_actor()

         
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        self.target_critic1 = self._build_critic()
        self.target_critic2 = self._build_critic()
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

         
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

         
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True, name='log_alpha')

         
        self.target_entropy = tf.constant(-float(action_dim), dtype=tf.float32)

        self.buffer = ReplayBuffer(buffer_capacity)

     
     
     

    def _build_actor(self):
        """Actor outputs [mean, log_std] concatenated — shape (batch, 2*action_dim)."""
        inputs = Input(shape=(self.state_dim,))
         
        x = Dense(self.actor_hidden, activation='tanh',
                  kernel_initializer=GlorotUniform())(inputs)
         
        outputs = Dense(2 * self.action_dim, activation=None,
                        kernel_initializer=GlorotUniform())(x)
        return Model(inputs, outputs)

    def _build_critic(self):
        """Critic Q(s,a) with linear output — negative losses are valid Q-values."""
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        concat = Concatenate()([state_input, action_input])
        x = Dense(self.critic_hidden, activation='relu',
                  kernel_initializer=HeNormal())(concat)
         
        outputs = Dense(1, activation=None,
                        kernel_initializer=HeNormal())(x)
        return Model([state_input, action_input], outputs)

     
     
     

    def _sample_action(self, state_batch, training=True):
        """
        Returns (action_01, log_pi) where action_01 ∈ [0,1]^action_dim.

        Uses reparameterisation trick:
          z = mean + std * eps,  eps ~ N(0,I)
          a_tanh = tanh(z)
          a_01   = (a_tanh + 1) / 2

        Log-prob includes tanh-squashing correction for valid density.
        """
        out = self.actor(state_batch, training=training)
        mean = out[:, :self.action_dim]
        log_std = tf.clip_by_value(out[:, self.action_dim:], LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        eps = tf.random.normal(tf.shape(mean))
        z = mean + std * eps

        action_tanh = tf.tanh(z)
        action_01 = (action_tanh + 1.0) / 2.0

         
        log_prob_z = -0.5 * (
            tf.square((z - mean) / (std + EPS))
            + 2.0 * log_std
            + tf.math.log(2.0 * np.pi)
        )
        log_prob_z = tf.reduce_sum(log_prob_z, axis=1, keepdims=True)

         
        tanh_correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.square(action_tanh) + EPS),
            axis=1, keepdims=True
        )

         
         
        scale_correction = tf.cast(self.action_dim, tf.float32) * tf.math.log(2.0)

        log_pi = log_prob_z - tanh_correction + scale_correction
        return action_01, log_pi

    def get_action_numpy(self, state_np):
        """Draw one action (numpy array) from current stochastic policy."""
        s = tf.convert_to_tensor(state_np[np.newaxis, :], dtype=tf.float32)
        action_01, _ = self._sample_action(s, training=False)
        return action_01[0].numpy()

     
     
     

    def map_continuous_action(self, raw_action):
        """Project action into the feasible budget set."""
        if self.player_role == 'defender':
             
             
            action = np.array(raw_action, dtype=np.float64)
            total = action.sum()
            if total > 1.0:
                action = action / total    
            return action

        elif self.player_role == 'adversary':
            if self.action_costs is None or self.budget is None:
                raise ValueError("Adversary requires action_costs and budget.")
            action_probs = np.array(raw_action)
            expected_cost = np.sum(action_probs * self.action_costs)
            if expected_cost > 0:
                scaling_factor = min(1.0, self.budget / expected_cost)
                return action_probs * scaling_factor
            return action_probs

     
     
     

    @tf.function
    def update_networks(self, state_batch, action_batch, reward_batch, next_state_batch):
        """
        One gradient step for critics, actor, and temperature.

        NaN guards:
          - log_std clamped in _sample_action
          - Q-targets clipped to [Q_CLIP_MIN, 0]
          - All gradient norms clipped to GRAD_CLIP_NORM
          - log_alpha clamped after each update
        """
        alpha = tf.exp(tf.clip_by_value(self.log_alpha, LOG_ALPHA_MIN, LOG_ALPHA_MAX))

         
        next_action, next_log_pi = self._sample_action(next_state_batch, training=False)

         
        with tf.GradientTape() as tape:
            nq1 = self.target_critic1([next_state_batch, next_action], training=False)
            nq2 = self.target_critic2([next_state_batch, next_action], training=False)
            next_q = tf.minimum(nq1, nq2) - alpha * next_log_pi

             
            y = tf.stop_gradient(
                tf.clip_by_value(
                    reward_batch + self.gamma * next_q,
                    Q_CLIP_MIN, Q_CLIP_MAX
                )
            )

            q1 = self.critic1([state_batch, action_batch], training=True)
            q2 = self.critic2([state_batch, action_batch], training=True)
            critic_loss = (tf.reduce_mean(tf.square(y - q1)) +
                           tf.reduce_mean(tf.square(y - q2)))

        all_critic_vars = (self.critic1.trainable_variables +
                           self.critic2.trainable_variables)
        critic_grads = tape.gradient(critic_loss, all_critic_vars)
         
        critic_grads = [g if g is not None else tf.zeros_like(v)
                        for g, v in zip(critic_grads, all_critic_vars)]
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, GRAD_CLIP_NORM)
        self.critic_optimizer.apply_gradients(zip(critic_grads, all_critic_vars))

         
        with tf.GradientTape() as tape:
            pi_action, log_pi = self._sample_action(state_batch, training=True)
            q1_pi = self.critic1([state_batch, pi_action], training=False)
            q2_pi = self.critic2([state_batch, pi_action], training=False)
            min_q_pi = tf.minimum(q1_pi, q2_pi)
             
            actor_loss = tf.reduce_mean(alpha * log_pi - min_q_pi)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads = [g if g is not None else tf.zeros_like(v)
                       for g, v in zip(actor_grads, self.actor.trainable_variables)]
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, GRAD_CLIP_NORM)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))

         
        with tf.GradientTape() as tape:
            _, log_pi_alpha = self._sample_action(state_batch, training=False)
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_pi_alpha + self.target_entropy)
            )

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        alpha_grads = [g if g is not None else tf.zeros_like(v)
                       for g, v in zip(alpha_grads, [self.log_alpha])]
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
         
        self.log_alpha.assign(tf.clip_by_value(self.log_alpha, LOG_ALPHA_MIN, LOG_ALPHA_MAX))

         
        for t_var, var in zip(self.target_critic1.variables, self.critic1.variables):
            t_var.assign(self.tau * var + (1.0 - self.tau) * t_var)
        for t_var, var in zip(self.target_critic2.variables, self.critic2.variables):
            t_var.assign(self.tau * var + (1.0 - self.tau) * t_var)

     
     
     

    def compute_best_response(self, env, opponent_pure_strategies,
                              opponent_mixed_strategy,
                              episodes=200, max_steps=100):
        """
        SAC-AP best-response oracle.

        No epsilon-greedy needed — SAC is inherently stochastic.
        Returns a callable policy(state) → numpy action array.
        """
        for episode in range(episodes):
            state = env.reset()

             
            sampled_idx = np.random.choice(
                len(opponent_pure_strategies), p=opponent_mixed_strategy)
            pi_minus_v = opponent_pure_strategies[sampled_idx]

            for step in range(max_steps):
                 
                raw_action = self.get_action_numpy(state)
                action = self.map_continuous_action(raw_action)

                 
                opponent_action = pi_minus_v(env.get_opponent_state())

                 
                if self.player_role == 'defender':
                    next_state, reward, done, _ = env.step(action, opponent_action)
                else:
                    next_state, def_reward, done, _ = env.step(opponent_action, action)
                    reward = -def_reward

                 
                self.buffer.record((state, raw_action, reward, next_state))

                if self.buffer.get_length() >= 256:
                    batch = self.buffer.sample_batch(256)
                    self.update_networks(*batch)

                state = next_state
                if done:
                    break

         
        def policy_fn(state):
            raw_action = self.get_action_numpy(np.asarray(state, dtype=np.float32))
            return self.map_continuous_action(raw_action)

        return policy_fn


class ReplayBuffer:
    """Circular replay buffer — identical contract to the one in DDPG.py."""

    def __init__(self, capacity=40000):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0

    def record(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience
        self.ptr = (self.ptr + 1) % self.capacity

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
         
         
        states = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
        actions = np.array([self.buffer[i][1] for i in indices], dtype=np.float32)
        rewards = np.array([[self.buffer[i][2]] for i in indices], dtype=np.float32)
        next_states = np.array([self.buffer[i][3] for i in indices], dtype=np.float32)

        return (
            tf.convert_to_tensor(states),
            tf.convert_to_tensor(actions),
            tf.convert_to_tensor(rewards),
            tf.convert_to_tensor(next_states)
        )

    def get_length(self):
        return len(self.buffer)
