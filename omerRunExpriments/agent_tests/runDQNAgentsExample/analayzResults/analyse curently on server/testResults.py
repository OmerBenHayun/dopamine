#matatn and omer code
import os
import pickle
import gin
import tensorflow as tf
import statistics
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rubust_dqn import rubust_dqn_agent
from dopamine.utils import example_viz_lib
import matplotlib.pyplot as plt

tmp_dir = os.path.join(os.path.expanduser('~'),'matan_omer','visualize_results','tmp')
output_dir = os.path.join(os.path.expanduser('~'),'matan_omer','visualize_results','output')
#output_dir = os.path.join(os.path.expanduser('~'),'dop_tmp','graph')


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
    ckpt_prefix_path = '/home/deep7/matan_omer/dopamine/tmp/dopamineSimpleDemoExpiriment/Asterix/ep_train_0.1/ep_eval_0.01'
    ckpt_list = [os.path.join(ckpt_prefix_path, file_path, 'ckpt.199') for file_path in
                 ['alpha_0/checkpoints', 'alpha_0.05/checkpoints', 'alpha_0.1/checkpoints']]
    alphas = ['alpha_0','alpha_0.05','alpha_0.1',]
    #res = dict()
    #res['alpha_0'] = os.path.join(ckpt_prefix_path, 'alpha_0/checkpoints', 'ckpt.199')
    #res['alpha_0.05'] = os.path.join(ckpt_prefix_path, 'alpha_0.05/checkpoints', 'ckpt.199')
    #res['alpha_0'] = os.path.join(ckpt_prefix_path, 'alpha_0/checkpoints', 'ckpt.199')
    res = dict(zip(alphas,ckpt_list))
    return res

def run_single_episode_and_return_reward(runner :example_viz_lib.MyRunner):
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

def get_average_episodes_reward_stats_from_runner(runner :example_viz_lib.MyRunner
                ,alpha_test : float,epsilon_test : float,num_of_episodes : int) -> (float,float):
    """
    return mean and variance from num_of_episodes episodes
    """
    runner._agent.alpha = alpha_test
    runner._agent.epsilon_eval = epsilon_test
    runner._agent.eval_mode = True
    rewards = []
    for i in range(num_of_episodes):
        rewards.append(run_single_episode_and_return_reward(runner))
    return statistics.mean(rewards) , statistics.stdev(rewards)

def get_average_episodes_reward_stats_from_ckpt_file(file_path :str,alpha_train_str : str,
                                                    alpha_test : float,epsilon_test : float,num_of_episodes : int) -> dict:
    tf.compat.v1.reset_default_graph()
    config = """
    atari_lib.create_atari_environment.game_name = '{}'
    OutOfGraphReplayBuffer.batch_size = 32
    OutOfGraphReplayBuffer.replay_capacity = 300
    """.format(GAME)
    #gin.bind_parameter('JaxRubustDQNAgent.alpha', alpha_test)
    #gin.bind_parameter('JaxRubustDQNAgent.epsi', alpha_test)
    gin.parse_config(config)

    runner = example_viz_lib.MyRunner(tmp_dir,file_path, create_rubust_dqn_agent)
    #model_name_str = "alpha_train_{}".format(alpha_train_str) #change here alpha
    mean , std =   get_average_episodes_reward_stats_from_runner(runner, alpha_test, epsilon_test, num_of_episodes)
    #return model_name_str, mean , variance
    #remove this lines only for debug
    #std = 0
    #mean = 0
    result = {'alpha_train' : alpha_train_str , 'mean' : mean , 'std' : std}
    return result

def compare_models_from_ckpt_files(alpha_train_to_ckpt_dict: dict,alpha_test : float,epsilon_test : float,num_of_episodes : int):
    #res = [] #each elment will be (model name , mean ,std) for the given hyperparamether and num_of_episodes.
    results= [] #alpha_train_to_ckpt_dict_list is a list of dics . each dic has single key - alpha_train and single value path to file.
    for alpha_train_str, file_path in alpha_train_to_ckpt_dict.items():
        results.append(get_average_episodes_reward_stats_from_ckpt_file(file_path,alpha_train_str,alpha_test, epsilon_test, num_of_episodes))
    create_graph_from_compression(results,alpha_test,epsilon_test,num_of_episodes,show_std = True)
    create_graph_from_compression(results,alpha_test,epsilon_test,num_of_episodes,show_std = False)

def create_graph_from_compression (results : list,alpha_test : float,epsilon_test : float ,num_of_episodes : int ,show_std: bool=False ):
    #results is a list of dicts
    alphas_train = [result['alpha_train'] for result in results]
    means = [result['mean'] for result in results]
    stds = [result['std'] for result in results]
    if (show_std):
        plt.bar(range(len(results)) ,means, align='center',yerr = stds)
    else:
        plt.bar(range(len(results)), means, align='center')

    plt.xticks(range(len(results)), alphas_train)
    plt.ylabel('Average reward per episode')
    plt.xlabel('Agent')
    title = 'Astrix - epsilon eval {} alpha test {} num of episodes {}'.format(str(epsilon_test),str(alpha_test),str(num_of_episodes))
    plt.title(title)
    plt.grid(True)
    file_name = os.path.join(output_dir,title.replace(' ','_'))
    print('yay')
    if show_std:
        file_name= file_name+ '_with_std'
    else:
        file_name= file_name+ '_without_std'
    print('save file' + file_name )
    plt.savefig(file_name,format='png')
    # Save the figure and show
    #plt.savefig('bar_plot_with_error_bars.png')
    plt.tight_layout()
    #plt.show()
    plt.clf()



GAME = 'Asterix'

def main():
    print("hi")
    #alpha_test = 0
    #epsilon_test =0
    num_of_episodes =80
    for epsilon_test in [0,0.01,0.05,0.1,0.15,0.2,0.5]:
        for alpha_test in [0,0.01,0.05,0.1,0.15,0.2,0.5]:
            alpha_train_to_ckpt_dict = alpha_train_to_ckpt()
            compare_models_from_ckpt_files(alpha_train_to_ckpt_dict, alpha_test, epsilon_test, num_of_episodes)

    print("bye")























if __name__ == '__main__':
    main()

