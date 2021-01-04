from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from absl import logging
import numpy as np

import os
from dopamine.jax.agents.dqn.dqn_agent import dqn_agent
from dopamine.jax.agents.rubust_dqn import rubust_dqn_agent as agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf

#THIS FILE IS STILL WIP - WORK IN PROGRESS.finish it when understandig how to unbundle model from checkpoint


#parameters:

#game parameters
GAMES = ['Asterix']
alpha_range = [0,0.05,0.1]
#alpha_range = [0.05]
epsilon_train_range = [0.1]
epsilon_eval_range = [0.01]
"""
#demo training parameters
Runner.num_iterations = 3
Runner.training_steps = 3  # agent steps
Runner.evaluation_steps = 3  # agent steps
Runner.max_steps_per_episode = 3  # agent steps
"""
# init demo params
# gin.bind_parameter('Runner.num_iterations',3)
# gin.bind_parameter('Runner.training_steps',3)
# gin.bind_parameter('Runner.evaluation_steps',3)
# gin.bind_parameter('Runner.max_steps_per_episode',3)
#original trainig parameters values:
"""
Runner.num_iterations = 200
Runner.training_steps = 250000  # agent steps
Runner.evaluation_steps = 125000  # agent steps
Runner.max_steps_per_episode = 27000  # agent steps
WrappedReplayBuffer.replay_capacity = 1000000
WrappedReplayBuffer.batch_size = 32
the game runned by the command:
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/dqn/configs/dqn.gin
on that settings the time took the model to train was around day and a half (around 35 hours)
the output dir called /tmp/dopamine_runs and it's size in the end was 1.1 GB
"""

#dir
baseDir = os.path.join('tmp','dopamineSimpleDemoExpiriment')

#replay buffer parameters
"""
WrappedReplayBuffer.replay_capacity = 1000000
WrappedReplayBuffer.batch_size = 32
"""


def create_experiment_log_path(base:str, game:str, ep_train:str, ep_eval:str, alpha:str) -> str:
    path = os.path.join(base,'game:'+game,'ep_train_'+ep_train,'ep_eval_'+ep_eval,'alpha_'+alpha)
    if not os.path.exists(path):
        os.makedirs(path)
    return path




def main():
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.disable_v2_behavior()
    with open(
            os.path.join(os.getcwd(),os.path.join('dopamine','jax','agents','rubust_dqn','configs','rubust_dqn.gin'))) as bindings:
        gin.parse_config(bindings)
    #start loop params
    for game in GAMES:
        gin.bind_parameter('atari_lib.create_atari_environment.game_name', game)
        for ep_train in epsilon_train_range:
            gin.bind_parameter('JaxDQNAgent.epsilon_train', ep_train)
            for ep_eval in epsilon_eval_range:
                gin.bind_parameter('JaxDQNAgent.epsilon_eval', ep_eval)
                for alpha in alpha_range:
                    gin.bind_parameter('JaxRubustDQNAgent.alpha', alpha)
                    LOG_PATH = create_experiment_log_path(baseDir, game,str(ep_train),str(ep_eval),str(alpha))
                    print('start experiment.game:{},epsilon train:{},epsilon_test:{},alpha robust:{}.'.format(game, str(ep_train), str(ep_eval), str(alpha)))
                    rubust_dqn_runner = run_experiment.Runner(LOG_PATH, agent.create_rubust_dqn_agent)         #with learning
                    rubust_dqn_runner.run_experiment()
                    print('finish experiment')

    print('finish all experiments\ndone')

if __name__ == '__main__':
    main()
