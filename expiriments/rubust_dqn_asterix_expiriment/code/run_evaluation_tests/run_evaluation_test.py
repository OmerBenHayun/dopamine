import os
import pickle
import gin
import tensorflow as tf
import statistics
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rubust_dqn import rubust_dqn_agent
from dopamine.utils import example_viz_lib
import matplotlib.pyplot as plt

#run this script from the root directory
#
#to run on server 
#activate venv
#make dopamine pythpon root dir
#go to dopamine main dir and run:
#python -m expiriments.rubust_dqn_asterix_expiriment.code.run_evaluation_tests.run_evaluation_test

base_path = os.path.join('expiriments','rubust_dqn_asterix_expiriment','code')
tmp_dir = os.path.join(base_path,'evaluation_output_tmp_dir')
output_dir = os.path.join(base_path,'evaluation_output_res_data')
trained_model_dir = os.path.join(base_path,'trained_agents_ckpt_files')

GAME = 'Asterix'
#epsilon_test_list = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5]
#alpha_test_list = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5]
#num_of_episodes = 100


#for debugging
epsilon_test_list = [0.01]
alpha_test_list = [0.01]
num_of_episodes = 2

output_res_name = 'evaluation_compression_num_of_episodes_{}.dict'.format(num_of_episodes)

# Customized JAX agent subclasses to record q-values, distributions, and  rewards.
class MyRubustDQNAgent(rubust_dqn_agent.JaxRubustDQNAgent):
    """Sample JAX DQN agent to visualize Q-values and rewards."""

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint):
        del use_legacy_checkpoint
        with open(checkpoint_path, 'rb') as f:
            bundle_dictionary = pickle.load(f)
        online_network = self.online_network.replace(
            params=bundle_dictionary['online_params'])
        optimizer_def = dqn_agent.create_optimizer('adam')
        self.optimizer = optimizer_def.create(online_network)
        self.target_network = self.target_network.replace(
            params=bundle_dictionary['target_params'])

def create_rubust_dqn_agent(sess, environment, summary_writer=None) ->rubust_dqn_agent.JaxRubustDQNAgent:
    del sess
    return MyRubustDQNAgent(num_actions=environment.action_space.n,
                      summary_writer=summary_writer)

def alpha_train_to_ckpt() -> dict:
    alpha_train_list = ['0','0.05','0.1']
    ckpt_list = [os.path.join(trained_model_dir,'ep_train_0.1_alpha_train_{}_ckpt.199'.format(alpha)) for alpha in alpha_train_list]
    alphas = ['alpha train: {}'.format(alpha) for alpha in alpha_train_list]
    res = dict(zip(alphas,ckpt_list))
    return res

def run_single_episode_and_return_reward(runner :example_viz_lib.MyRunner) -> float:
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

def get_episodes_rewards_from_runner(runner :example_viz_lib.MyRunner
                ,alpha_test : float,epsilon_test : float,num_of_episodes : int) -> list:
    runner._agent.alpha = alpha_test
    runner._agent.epsilon_eval = epsilon_test
    runner._agent.eval_mode = True
    rewards = []
    for i in range(num_of_episodes):
        rewards.append(run_single_episode_and_return_reward(runner))
    return rewards

def get_episodes_rewards_from_ckpt_file(file_path :str,alpha_train_str : str,
                                                    alpha_test : float,epsilon_test : float,num_of_episodes : int) -> list:
    tf.compat.v1.reset_default_graph()
    config = """
    atari_lib.create_atari_environment.game_name = '{}'
    OutOfGraphReplayBuffer.batch_size = 32
    OutOfGraphReplayBuffer.replay_capacity = 300
    """.format(GAME)
    gin.parse_config(config)

    runner = example_viz_lib.MyRunner(tmp_dir,file_path, create_rubust_dqn_agent)
    return get_episodes_rewards_from_runner(runner, alpha_test, epsilon_test, num_of_episodes)

def get_episodes_rewards_from_ckpt_files_for_given_hyperparameters(alpha_train_to_ckpt_dict: dict,alpha_test : float,epsilon_test : float,num_of_episodes : int) -> dict:
    #hyperparameters = {'alpha test' : alpha_test , 'epsilon test' : epsilon_test}
    res = dict()
    for alpha_train_str, file_path in alpha_train_to_ckpt_dict.items():
         res[alpha_train_str] = (get_episodes_rewards_from_ckpt_file(file_path,alpha_train_str,alpha_test, epsilon_test, num_of_episodes))
    return res


def main():
    print("hi")
    alpha_train_to_ckpt_dict = alpha_train_to_ckpt()
    results = dict()
    for epsilon_test in epsilon_test_list:
        for alpha_test in alpha_test_list:
            #hyperparameters = {'alpha test': alpha_test, 'epsilon test': epsilon_test}
            hyperparameters = ( 'alpha test: {}'.format(alpha_test),'epsilon test: {}'.format(epsilon_test) )
            results[hyperparameters] = get_episodes_rewards_from_ckpt_files_for_given_hyperparameters(alpha_train_to_ckpt_dict, alpha_test, epsilon_test, num_of_episodes)

    #save results
    with open(os.path.join(output_dir,output_res_name) , 'wb') as handle:
        pickle.dump(results,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print("bye")




if __name__ == '__main__':
    main()
