from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
from dopamine.jax.agents.dqn.dqn_agent import dqn_agent
from dopamine.jax.agents.rubust_dqn import rubust_dqn_agent as agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf


#parameters:
GAMES = ['Asterix']
alpha_range = [0,0.05,0.1]
epsilon_train_range = [0.1]
epsilon_eval_range = [0.01]


def create_experiment_log_path(base:str, game:str, ep_train:str, ep_eval:str, alpha:str) -> str:
    path = os.path.join(base,game,'ep_train_'+ep_train,'ep_eval_'+ep_eval,'alpha_'+alpha)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def main():
    with open(
            os.path.join(os.getcwd(),os.path.join('dopamine','jax','agents','rubust_dqn','configs','rubust_dqn.gin'))) as bindings:
        gin.parse_config(bindings)
    for game in GAMES:
        gin.bind_parameter('atari_lib.create_atari_environment.game_name', game)
        for ep_train in epsilon_train_range:
            gin.bind_parameter('JaxDQNAgent.epsilon_train', ep_train)
            for ep_eval in epsilon_eval_range:
                gin.bind_parameter('JaxDQNAgent.epsilon_eval', ep_eval)
                for alpha in alpha_range:
                    gin.bind_parameter('JaxRubustDQNAgent.alpha', alpha)
                    LOG_PATH = create_experiment_log_path(os.getcwd(), game, str(ep_train), str(ep_eval), str(alpha))
                    print('start experiment.game:{},epsilon train:{},epsilon_test:{},alpha robust:{}.'.format(game, str(ep_train), str(ep_eval), str(alpha)))
                    dqn_runner = run_experiment.Runner(LOG_PATH, agent.create_rubust_dqn_agent)         #with learning
                    dqn_runner.run_experiment()
                    print('finish experiment')

    print('finish all experiments\ndone')

if __name__ == '__main__':
    main()

