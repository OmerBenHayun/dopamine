import expiriments.rubust_dqn_asterix_expiriment.code.graph_creator_from_results.utils as loader
import statistics
import matplotlib.pyplot as plt
import os


output_graph_path = os.path.join('expiriments','rubust_dqn_asterix_expiriment','results','bar_plots')

def save_bar_plot(title :str ,input_to_graph: dict,show_std :bool):
    mean_list = []
    std_list = []
    alpha_train_list = []
    for alpha_train , results in input_to_graph.items():
       alpha_train_list.append(alpha_train)
       mean_list.append(statistics.mean(results))
       std_list.append(statistics.stdev(results))
    #alphas_train = [result['alpha_train'] for result in results]
    #means = [result['mean'] for result in results]
    #stds = [result['std'] for result in results]
    if (show_std):
        plt.bar(range(len(mean_list)) ,mean_list, align='center',yerr = std_list)
    else:
        plt.bar(range(len(mean_list)), mean_list, align='center')

    plt.xticks(range(len(alpha_train_list)), alpha_train_list)
    plt.ylabel('Average reward per episode')
    plt.xlabel('Agent')
    #title = 'Astrix - epsilon eval {} alpha test {} num of episodes {}'.format(str(epsilon_test),str(alpha_test),str(num_of_episodes))
    plt.title(title)
    plt.grid(True)
    file_name = os.path.join(output_graph_path,title.replace(': ','_'))
    file_name = file_name.replace('. ','_')
    file_name = file_name.replace(' ','_')
    print('yay')
    if show_std:
        file_name= file_name+ '_with_std'
    else:
        file_name= file_name+ '_without_std'
    print('save file' + file_name )
    plt.savefig(file_name,format='png')
    # Save the figure and show
    plt.tight_layout()
    #plt.show()
    plt.clf()

def main():
   num_of_episodes  = loader.get_num_of_episodes()
   results = loader.load_results()
   for k,v in results.items():
       save_bar_plot('Game: Astrix. '+'num of episodes: {}. '.format(num_of_episodes) +k[0]+'. '+k[1] ,v,True)

if __name__ == '__main__':
    print("start")
    main()
    print("finish")
