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

def list_of_pkl_files() -> dict:
    ckpt_prefix_path = '/media/omer/3264-3930/OneDrive - Technion/Projects/ProjectA/dopamine/firstResultsOfExpirement/tmp/dopamineSimpleDemoExpiriment/Asterix/ep_train_0.1/ep_eval_0.01'
    ckpt_list = [os.path.join(ckpt_prefix_path, file_path, 'ckpt.199') for file_path in
                 ['alpha_0/checkpoints', 'alpha_0.05/checkpoints', 'alpha_0.1/checkpoints']]
    #res = dict()
    #res['alpha_0'] = os.path.join(ckpt_prefix_path, 'alpha_0/checkpoints', 'ckpt.199')
    #res['alpha_0.05'] = os.path.join(ckpt_prefix_path, 'alpha_0.05/checkpoints', 'ckpt.199')
    #res['alpha_0'] = os.path.join(ckpt_prefix_path, 'alpha_0/checkpoints', 'ckpt.199')
    return ckpt_list

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
    return statistics.mean(rewards) , statistics.variance(rewards)

def get_average_episodes_reward_stats_from_ckpt_file(file_path :str,
                                                    alpha_test : float,epsilon_test : float,num_of_episodes : int) -> (str,float,float):
    tf.compat.v1.reset_default_graph()
    config = """
    atari_lib.create_atari_environment.game_name = '{}'
    OutOfGraphReplayBuffer.batch_size = 32
    OutOfGraphReplayBuffer.replay_capacity = 300
    """.format(GAME)
    gin.parse_config(config)
    runner = example_viz_lib.MyRunner('',file_path, create_rubust_dqn_agent)
    model_name_str = "alpha_train_{}".format(runner._agent.alpha)
    mean , variance =   get_average_episodes_reward_stats_from_runner(runner, alpha_test, epsilon_test, num_of_episodes)
    return model_name_str, mean , variance

def compare_models_from_ckpt_files(ckpt_files : list,alpha_test : float,epsilon_test : float,num_of_episodes : int):
    res = [] #each elment will be (model name , mean ,std) for the given hyperparamether and num_of_episodes.
    for file_path in ckpt_files:
        res.append(get_average_episodes_reward_stats_from_ckpt_file(file_path, alpha_test, epsilon_test, num_of_episodes))

def create_graph_from_compression (data : list,alpha_test : float,epsilon_test : float ,num_of_episodes : int ,show_std : bool):
    #each element on list is tuple with three elements (str, float, float)
    name_alpha_train , mean , std = zip(*data)
    if (show_std):
        plt.bar(range(len(data)) ,mean, align='center',yerr = std)
    else:
        plt.bar(range(len(data)), mean, align='center')

    plt.xticks(range(len(data)), name_alpha_train)
    plt.ylabel('Average reward per episode')
    plt.xlabel('Agent')
    title = "game astrix - epsilon eval {}, alpha test {}, num_of_episodes {}".format(epsilon_test,alpha_test,num_of_episodes)
    plt.title(title)
    plt.grid(True)
    file_name = title.replace(title,"_")
    plt.savefig(file_name)
    # Save the figure and show
    #plt.savefig('bar_plot_with_error_bars.png')
    plt.tight_layout()
    plt.show()



GAME = 'Asterix'

def main():
    print("hi")
    ##tf.compat.v1.reset_default_graph()
    ##config = """
    ##atari_lib.create_atari_environment.game_name = '{}'
    ##OutOfGraphReplayBuffer.batch_size = 32
    ##OutOfGraphReplayBuffer.replay_capacity = 300
    ##""".format(GAME)
    ##gin.parse_config(config)


    #first_ckpt_file = list_of_pkl_files()[0]
    ##runner = example_viz_lib.MyRunner('',first_ckpt_file, create_rubust_dqn_agent)
    #mean, std = get_average_episodes_reward_stats_from_runner(runner,0,0,2)
    alpha_test = 0
    epsilon_test =0
    num_of_episodes =2
    res = []
    for file_path in list_of_pkl_files():
        res.append(get_average_episodes_reward_stats_from_ckpt_file(file_path , alpha_test , epsilon_test , num_of_episodes ))
    #create_graph_from_compression([("a",1,2),("b",3,4),("c",5,6)],alpha_test,epsilon_test,num_of_episodes)
    create_graph_from_compression(res,alpha_test,epsilon_test,num_of_episodes,show_std=False)
    print("bye")


















if __name__ == '__main__':
    main()

