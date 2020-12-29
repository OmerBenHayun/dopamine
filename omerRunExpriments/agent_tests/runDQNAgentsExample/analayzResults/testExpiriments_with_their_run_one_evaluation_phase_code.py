import collections
import dopamine.colab.utils as colab_utils

from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rubust_dqn import rubust_dqn_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.replay_memory import circular_replay_buffer
from dopamine.utils import example_viz_lib
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import os
import os.path as osp
import pathlib
import pickle
import tensorflow as tf
#from omerRunExpriments.agent_tests.runDQNAgentsExample.RunRubustDQNAgentSingleGameTest.runExpiriments import create_experiment_log_path
import statistics
import matplotlib.pyplot as plt
# Customized JAX agent subclasses to record q-values, distributions, and  rewards.
class MyDQNAgent(rubust_dqn_agent.JaxRubustDQNAgent):
    """Sample JAX DQN agent to visualize Q-values and rewards."""

    def __init__(self, num_actions, summary_writer=None):
        super().__init__(num_actions, summary_writer=summary_writer)
        self.q_values = [[] for _ in range(num_actions)]
        self.rewards = []

    def _record_q_values(self):
        q_values = self.online_network(self.state).q_values
        for i, q_value in enumerate(q_values):
            self.q_values[i].append(q_value)

    def step(self, reward, observation):
        action = super().step(reward, observation)
        self.rewards.append(reward)
        self._record_q_values()
        return action

    def begin_episode(self, observation):
        action = super().begin_episode(observation)
        self._record_q_values()
        return action

    def end_episode(self, reward):
        super().end_episode(reward)
        self.rewards.append(reward)
        self._record_q_values()

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint):
        del use_legacy_checkpoint
        with open(checkpoint_path, 'rb') as f:
            bundle_dictionary = pickle.load(f)
        online_network = self.online_network.replace(
            params=bundle_dictionary['online_params'])
        """
        optimizer_def = dqn_agent.create_optimizer('adam')
        self.optimizer = optimizer_def.create(online_network)
        self.target_network = self.target_network.replace(
            params=bundle_dictionary['target_params'])
        """

    def get_q_values(self):
        return self.q_values

    def get_rewards(self):
        return [onp.cumsum(self.rewards)]


def create_dqn_agent(sess, environment, summary_writer=None):
    del sess
    return MyDQNAgent(num_actions=environment.action_space.n,
                      summary_writer=summary_writer)






OUTPUT_PATH = '/tmp/output_video_path'
BASE_PATH = '/tmp/colab_dope_run'  # @param
GAMES = [ 'Asterix']  # @param


#dir
baseDir = os.path.join('tmp','dopamineSimpleDemoExpiriment')

def RunEpsodeReturnReward(runner)->float:
    accumulated_reward = 0
    initial_observation = runner._environment.reset()
    action = runner._agent.begin_episode(initial_observation)
    while True:
        observation, reward, is_terminal, _ = runner._environment.step(action)
        # reward_list.append(reward)
        accumulated_reward += reward
        if runner._environment.game_over:
            break
        elif is_terminal:
            runner._agent.end_episode(reward)
        else:
            action = runner._agent.step(reward, observation)
    runner._end_episode(reward)
    return accumulated_reward

def plotRes(dict):
    #fig, ax = plt.subplots()
    plt.bar(range(len(dict)), [statistics.mean(i) for i in dict.values()], align='center')#,
           #yerr=  [statistics.variance(i) for i in dict.values()])
    plt.xticks(range(len(dict)), list(dict.keys()))
    plt.ylabel('average reward per episode')
    plt.xlabel('agent')
    plt.savefig('bar plot.png')
    plt.title("Game Astrix - epsilon eval 0.01, alpha test 0")
    plt.grid(True)
    # Save the figure and show
    plt.tight_layout()
    #plt.savefig('bar_plot_with_error_bars.png')
    plt.show()

def main2():
    print("start")
    game = 'Asterix'
    num_steps = 200  # @param {type:'slider', min:50, max:500}
    tf.compat.v1.reset_default_graph()
    config = """
    Runner.num_iterations = 3
    Runner.training_steps = 3
    Runner.evaluation_steps = 3
    Runner.max_steps_per_episode = 3
    atari_lib.create_atari_environment.game_name = '{}'
    OutOfGraphReplayBuffer.batch_size = 32
    OutOfGraphReplayBuffer.replay_capacity = 300
    OutOfGraphPrioritizedReplayBuffer.batch_size = 32
    OutOfGraphPrioritizedReplayBuffer.replay_capacity = 300
    """.format(game)
    base_dir = pathlib.Path('/tmp/agent_viz') / game / 'r_dqn'

    ckpt_prefix_path = '/media/omer/3264-3930/OneDrive - Technion/Projects/ProjectA/dopamine/firstResultsOfExpirement/tmp/dopamineSimpleDemoExpiriment/Asterix/ep_train_0.1/ep_eval_0.01'
    ckpt_list = [os.path.join(ckpt_prefix_path,file_path,'ckpt.199') for file_path in ['alpha_0/checkpoints','alpha_0.05/checkpoints','alpha_0.1/checkpoints']]

    gin.parse_config(config)
    """
    runner = example_viz_lib.MyRunner(base_dir,
                                      '/media/omer/3264-3930/OneDrive - Technion/Projects/ProjectA/dopamine/resultsOfSmallExpiriment/ckpt.199',
                                      create_dqn_agent)
    """
    res_dict ={}
    for ckpt,alpha in zip(ckpt_list,[0,0.05,0.1]):
        runner = example_viz_lib.MyRunner(base_dir,ckpt,create_dqn_agent)
        runner._agent.alpha =0
        runner._agent.epsilon_eval = 0.1
        runner._agent.eval_mode = True
        accumulated_reward_list = []
        for i in range(30):
            accumulated_reward_list.append(RunEpsodeReturnReward(runner))
        res_dict['alpha_{}'.format(alpha)] = accumulated_reward_list
    plotRes(res_dict)
    print("end")

if __name__ == '__main__':
    main2()


