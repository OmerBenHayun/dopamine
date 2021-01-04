"""modified version of DQN  of a DQN agent in JAx:rubust DQN agent"""

import numpy as np
import os
from dopamine.jax.agents.dqn import dqn_agent
from absl import logging
import functools
import jax
import jax.numpy as jnp
import numpy as onp
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import tensorflow as tf
import gin.tf



#BASE_PATH = '/tmp/colab_dope_run' todo:remove this line
#GAME = 'Asterix'todo:remove this line
#LOG_PATH = os.path.join(BASE_PATH, 'random_dqn', GAME) todo:remove this line



@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 9, 10, 11))
def select_action(network, state, rng, num_actions, eval_mode,
                         epsilon_eval, epsilon_train, epsilon_decay_period,
                         training_steps, min_replay_history, epsilon_fn,alpha):
    """Select an action (using rubust method) from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise,
         with probability of (1-alpha) acts greedily according to the current Q-value estimates.(maximaizer)
         in any other case choose the worst action acts adversary greedily according to the current Q-value estimates.(minimizer)

    Args:
      network: Jax Module to use for inference.
      state: input state to use for inference.
      rng: Jax random number generator.
      num_actions: int, number of actions (static_argnum).
      eval_mode: bool, whether we are in eval mode (static_argnum).
      epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
      epsilon_train: float, epsilon value to use in train mode (static_argnum).
      epsilon_decay_period: float, decay period for epsilon value for certain
        epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
      training_steps: int, number of training steps so far.
      min_replay_history: int, minimum number of steps in replay buffer
        (static_argnum).
      epsilon_fn: function used to calculate epsilon value (static_argnum).
      alpha: alpha robust parameter (static_argnum).

    Returns:
      rng: Jax random number generator.
      action: int, the selected action.
    """
    epsilon = jnp.where(eval_mode,
                        epsilon_eval,
                        epsilon_fn(epsilon_decay_period,
                                   training_steps,
                                   min_replay_history,
                                   epsilon_train))

    rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
    p1 = jax.random.uniform(rng1)
    p2 = jax.random.uniform(rng2)
    return rng, jnp.where(p1 <= epsilon,
                          jax.random.randint(rng2, (), 0, num_actions),
                          jnp.where(p2 <= alpha,
                                    jnp.argmin(network(state).q_values, axis=1)[0],
                                    jnp.argmax(network(state).q_values, axis=1)[0]))


@functools.partial(jax.jit, static_argnums=(8))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, cumulative_gamma, alpha):
  """Run the training step."""
  def loss_fn(model, target):
    q_values = jax.vmap(model, in_axes=(0))(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    loss = jnp.mean(jax.vmap(dqn_agent.huber_loss)(target, replay_chosen_q))
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  target = target_q(target_network,
                    next_states,
                    rewards,
                    terminals,
                    cumulative_gamma,
                    alpha)
  loss, grad = grad_fn(optimizer.target, target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


def target_q(target_network, next_states, rewards, terminals, cumulative_gamma, alpha):
  """Compute the target Q-value."""
  q_vals = jax.vmap(target_network, in_axes=(0))(next_states).q_values
  q_vals = jnp.squeeze(q_vals)
  replay_next_qt_max = jnp.max(q_vals, 1)
  replay_next_qt_min = jnp.min(q_vals, 1)
  # Calculate the Bellman target value according to robust strategy.
  #   Q_t = R_t + \gamma^N * Q'_t+1
  # where,
  #   Q'_t+1 = (1-\alpha)*(\argmax_a Q(S_t+1, a))+\alpha*\argmin_a Q(S_t+1, a)
  #          (or) 0 if S_t is a terminal state,
  # and
  #   N is the update horizon (by default, N=1).
  return jax.lax.stop_gradient(rewards + cumulative_gamma * (1. - terminals) *
                               ((1-alpha)*replay_next_qt_max + alpha * replay_next_qt_min))




@gin.configurable
class JaxRubustDQNAgent(dqn_agent.JaxDQNAgent):
    def __init__(self,num_actions,summary_writer=None,alpha = 0.1):
        """This maintains all the DQN default argument values"""
        super().__init__(num_actions,summary_writer=summary_writer)
        logging.info('\t alpha: %f', alpha)
        self.alpha = alpha

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """
        self._reset_state()
        self._record_observation(observation)

        if not self.eval_mode:
            self._train_step()

        self._rng, self.action = select_action(self.online_network,
                                               self.state,
                                               self._rng,
                                               self.num_actions,
                                               self.eval_mode,
                                               self.epsilon_eval,
                                               self.epsilon_train,
                                               self.epsilon_decay_period,
                                               self.training_steps,
                                               self.min_replay_history,
                                               self.epsilon_fn,
                                               self.alpha)
        self.action = onp.asarray(self.action)
        return self.action
    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.

        Returns:
          int, the selected action.
        """
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward, False)
            self._train_step()

        self._rng, self.action = select_action(self.online_network,
                                               self.state,
                                               self._rng,
                                               self.num_actions,
                                               self.eval_mode,
                                               self.epsilon_eval,
                                               self.epsilon_train,
                                               self.epsilon_decay_period,
                                               self.training_steps,
                                               self.min_replay_history,
                                               self.epsilon_fn,
                                               self.alpha)
        self.action = onp.asarray(self.action)
        return self.action

    def _train_step(self):
        """Runs a single training step.

        Runs training if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_network to target_network if training steps
        is a multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                self._sample_from_replay_buffer()
                self.optimizer, loss = train(self.target_network,
                                             self.optimizer,
                                             self.replay_elements['state'],
                                             self.replay_elements['action'],
                                             self.replay_elements['next_state'],
                                             self.replay_elements['reward'],
                                             self.replay_elements['terminal'],
                                             self.cumulative_gamma,
                                             self.alpha)
                if (self.summary_writer is not None and
                        self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = tf.compat.v1.Summary(value=[
                        tf.compat.v1.Summary.Value(tag='HuberLoss', simple_value=loss)])
                    self.summary_writer.add_summary(summary, self.training_steps)
            if self.training_steps % self.target_update_period == 0:
                self._sync_weights()

        self.training_steps += 1


def create_rubust_dqn_agent(sess,environment, summary_writer=None):
  """The Runner class will expect a function of this type to create an agent."""
  return JaxRubustDQNAgent(num_actions=environment.action_space.n)



