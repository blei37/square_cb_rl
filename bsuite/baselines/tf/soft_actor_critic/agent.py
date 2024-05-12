# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple implementation of Bootstrapped DQN with prior networks.

References:
1. "Deep Exploration via Bootstrapped DQN" (Osband et al., 2016)
2. "Deep Exploration via Randomized Value Functions" (Osband et al., 2017)
3. "Randomized Prior Functions for Deep RL" (Osband et al, 2018)

Links:
1. https://arxiv.org/abs/1602.04621
2. https://arxiv.org/abs/1703.07608
3. https://arxiv.org/abs/1806.03335

Notes:

- This agent is implemented with TensorFlow 2 and Sonnet 2. For installation
  instructions for these libraries, see the README.md in the parent folder.
- This implementation is potentially inefficient, as it does not parallelise
  computation across the ensemble for simplicity and readability.
"""

import copy
from typing import Callable, NamedTuple, Optional, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree



class SAC(base.Agent):
    """Soft Actor-Critic (SAC) agent."""

    def __init__(
      self,
      action_spec: specs.BoundedArray,
      actor_network: snt.Module,
      q1_network: snt.Module,
      q2_network: snt.Module,
      batch_size: int,
      discount: float,
      critic_learning_rate: float,
      actor_learning_rate: float,
      alpha_learning_rate: float,
      target_update_period: int,
      tau: float,
      seed: Optional[int] = None,
    ):
      # Internalise hyperparameters.
      self._num_actions = action_spec.num_values
      self._discount = discount
      self._batch_size = batch_size
      self._target_update_period = target_update_period
      self._tau = tau

      # Seed the RNG.
      tf.random.set_seed(seed)
      self._rng = np.random.RandomState(seed)

      # Internalise the components (networks, optimizer).
      self._actor_network = actor_network
      self._actor_optimizer = snt.optimizers.Adam(learning_rate=actor_learning_rate)
      self._q1_network = q1_network
      self._q2_network = q2_network
      self._critic_optimizer = snt.optimizers.Adam(learning_rate=critic_learning_rate)
      self._target_q1_network = copy.deepcopy(q1_network)
      self._target_q2_network = copy.deepcopy(q2_network)

      # Entropy-related
      self._target_entropy = -self._num_actions
      self._log_alpha = tf.Variable(0.0, trainable=True)
      self._alpha_optimizer = snt.optimizers.Adam(learning_rate=alpha_learning_rate)

      self._total_steps = tf.Variable(0)

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
      # Extract observation from the timestep
      observation = tf.convert_to_tensor(timestep.observation[None, ...])
      # print(observation)
      # print(observation.shape)
      # Add batch dimension if needed
      # if len(observation.shape) == 1:
      #     observation = tf.expand_dims(observation, axis=0)

      # Sample action from the policy.
      # action_mean, action_log_std = self._actor_network(observation)
      # Forward pass through the actor network
      print(observation)
      print(observation.shape)
      action_concat = self._actor_network(observation)
      
      # Split the concatenated output into mean and log_std tensors
      action_mean, action_log_std = tf.split(action_concat, num_or_size_splits=2, axis=-1)
    
    # Sample action from the mean and log_std
      action_std = tf.exp(action_log_std)
      raw_action = action_mean + action_std * tf.random.normal(tf.shape(action_mean))
      clipped_action = tf.clip_by_value(raw_action, -1, 1)  # assuming action range is [-1, 1]
      print(clipped_action)
      action = clipped_action[0, 0] 
      print(action)
      return int(action)

    def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
    ):
      self._total_steps.assign_add(1)
      observations = timestep.observation
      next_observations = new_timestep.observation
      rewards = new_timestep.reward
      discounts = new_timestep.discount

      # Update critic networks
      with tf.GradientTape(persistent=True) as tape:
        # Sample actions from the actor network for next observations
        # target_actions, target_log_probs = self._actor_network.sample(next_observations)
        # Assuming self._actor_network outputs action_mean and action_log_std
        print(next_observations)
        action_mean, action_log_std = self._actor_network(next_observations)

        # Create a Gaussian distribution based on the mean and log_std
        action_distribution = tf.distributions.Normal(action_mean, tf.exp(action_log_std))

        # Sample actions from the distribution
        target_actions = action_distribution.sample()

        # Calculate log probabilities of the sampled actions
        target_log_probs = action_distribution.log_prob(target_actions)
        target_q1_values = self._target_q1_network([next_observations, target_actions])
        target_q2_values = self._target_q2_network([next_observations, target_actions])
        # Use minimum Q-value between two target Q-functions
        target_values = tf.minimum(target_q1_values, target_q2_values) - tf.exp(self._log_alpha) * target_log_probs
        target_returns = rewards + discounts * self._discount * target_values

        # Compute current Q-values
        current_q1_values = self._q1_network([observations, action])
        current_q2_values = self._q2_network([observations, action])
        # Compute critic loss
        critic_loss = 0.5 * (
            tf.reduce_mean((current_q1_values - target_returns) ** 2) +
            tf.reduce_mean((current_q2_values - target_returns) ** 2)
        )

      # Compute gradients and update critic networks
      critic_gradients = tape.gradient(critic_loss, self._q1_network.trainable_variables +
                                        self._q2_network.trainable_variables)
      self._critic_optimizer.apply(critic_gradients, self._q1_network.trainable_variables +
                                    self._q2_network.trainable_variables)

      # Update actor network
      with tf.GradientTape() as tape:
        # Sample actions and log probabilities from the actor network
        new_actions, log_probs = self._actor_network.sample(observations)
        q1_values = self._q1_network([observations, new_actions])
        # Compute actor loss
        actor_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_probs - q1_values)

        # Compute gradients and update actor network
        actor_gradients = tape.gradient(actor_loss, self._actor_network.trainable_variables)
        self._actor_optimizer.apply(actor_gradients, self._actor_network.trainable_variables)

      # Update temperature parameter alpha
      with tf.GradientTape() as tape:
        alpha_loss = -tf.reduce_mean(self._log_alpha * (tf.stop_gradient(log_probs) + self._target_entropy))

      alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
      self._alpha_optimizer.apply(alpha_gradients, [self._log_alpha])

      # Update target critic networks
      if tf.math.mod(self._total_steps, self._target_update_period) == 0:
        self._update_target_networks()

    def _update_target_networks(self):
      # Update target Q-functions by soft update
      for target_param, param in zip(self._target_q1_network.trainable_variables +
                                      self._target_q2_network.trainable_variables,
                                      self._q1_network.trainable_variables + self._q2_network.trainable_variables):
        target_param.assign(self._tau * param + (1 - self._tau) * target_param)


def default_agent(obs_spec: specs.Array,
                      action_spec: specs.BoundedArray):
  """Initialize a SAC agent with default parameters."""
  # Define actor network
  hidden_units = (256, 256)
  actor_network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50,50, action_spec.num_values]),
        # snt.Linear(2),
    ])
  # Define two Q-networks
  q1_network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50, 50, action_spec.num_values]),
        # snt.Linear(1),  # Q-value output
    ])
  q2_network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50, 50, action_spec.num_values]),
        snt.Linear(1),  # Q-value output
    ])
  return SAC(
    action_spec=action_spec,
    actor_network=actor_network,
    q1_network=q1_network,
    q2_network=q2_network,
    batch_size=256,
    discount=0.99,
    critic_learning_rate=3e-4,
    actor_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    target_update_period=1,
    tau=5e-3,
    seed=42
  )